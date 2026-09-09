'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  MKRN3:  '#4a148c',  // deep purple — imprinting/ubiquitin; most common hereditary CPP
  DLK1:   '#1565c0',  // deep blue — imprinting cluster 14q32; 2nd most common
  KISS1:  '#e65100',  // deep amber — kisspeptin ligand GOF; extreme early onset
  KISS1R: '#880e4f',  // dark pink — GPR54 GOF; same gene as IHH
  LIN28B: '#1b5e20',  // dark green — RNA binding let-7 GWAS
  GNAS:   '#bf360c',  // deep red — MAS somatic; peripheral PP; GnRHa fails
  LEPR:   '#37474f',  // slate — leptin receptor; obesity + CPP rebound
  GNRH1:  '#006064',  // dark teal — GnRH activating; rare ~3-5%
};

const GENE_DISEASE = {
  MKRN3:  'CPP2 (AD paternal imprint LOF) — MOST COMMON hereditary CPP (~46% familial); X-linked-like: only paternal LOF symptomatic; GnRHa curative',
  DLK1:   'CPP (AD maternal LOF) — 2nd most common (~16% familial); paternally expressed gene; maternal LOF → precocious GnRH axis; GnRHa effective',
  KISS1:  'CPP1 GOF (AR/AD) — kisspeptin-1 ligand GOF; R73C resists MMP cleavage → prolonged activity; extreme early onset <4 yr; GnRHa curative',
  KISS1R: 'CPP1 GOF (AD) — GPR54 constitutive activation (A243V); same gene as IHH (LOF) — OPPOSITE phenotypes; GnRHa curative; earliest onset ~1 yr',
  LIN28B: 'CPP (AD GOF) — let-7 miRNA repressor; derepresses KISS1R/GNRH1; strongest GWAS locus female puberty timing; GnRHa curative',
  GNAS:   'McCune-Albright MAS (somatic GOF) — PERIPHERAL precocious puberty; GnRHa FAILS; aromatase inhibitor (letrozole) Rx; café-au-lait Coast-of-Maine PATHOGNOMONIC',
  LEPR:   'LEPR deficiency (AR LOF) — severe early-onset hyperphagia + obesity; absent puberty; CPP onset as rebound on metreleptin/weight-loss; GnRHa +/- metreleptin',
  GNRH1:  'CPP (AD activating) — GnRH hyperpulse; rare ~3-5% familial CPP; same gene as isolated IHH (LOF) without anosmia; GnRHa curative',
};

const INHERITANCE = {
  MKRN3: 'AD-paternal-imprint', DLK1: 'AD-maternal-LOF', KISS1: 'AR/AD-GOF',
  KISS1R: 'AD-GOF', LIN28B: 'AD-GOF', GNAS: 'Somatic-GOF', LEPR: 'AR-LOF', GNRH1: 'AD-GOF',
};

const CPP_GROUP = {
  MKRN3:  'Imprinting / Ubiquitin (Xq27.1)',
  DLK1:   'Imprinting / Notch-Astrocyte (14q32.2)',
  KISS1:  'Kisspeptin Ligand GOF (1q32.1)',
  KISS1R: 'GPR54 Receptor GOF (19p13.3)',
  LIN28B: 'RNA-Binding / let-7 (6q16.3)',
  GNAS:   'Somatic Gsα / McCune-Albright (20q13.32)',
  LEPR:   'Leptin Receptor / Metabolic (1p31.3)',
  GNRH1:  'GnRH Neuropeptide GOF (8p21.2)',
};

export default function HereditaryPrecociousPubertyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedGene, setSelectedGene] = useState('MKRN3');

  useEffect(() => {
    async function load() {
      try {
        const [ov, bk, df] = await Promise.all([
          fetch(`${API}/api/hereditary-precocious-puberty-atlas/overview`).then(r => r.json()),
          fetch(`${API}/api/hereditary-precocious-puberty-atlas/breakdown`).then(r => r.json()),
          fetch(`${API}/api/hereditary-precocious-puberty-atlas/definitions`).then(r => r.json()),
        ]);
        setOverview(ov); setBreakdown(bk); setDefinitions(df);
      } catch (e) { setError(e.message); }
      finally { setLoading(false); }
    }
    load();
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#4a148c' }}>Loading Hereditary Precocious Puberty Atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const accentColor = '#4a148c';
  const genes = Object.keys(GENE_COLORS);
  const gd = breakdown?.breakdown_by_gene?.[selectedGene];

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', maxWidth: 1200, margin: '0 auto', padding: 20 }}>
      {/* Header */}
      <div style={{ background: `linear-gradient(135deg,${accentColor} 0%,#880e4f 100%)`, borderRadius: 12, padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 700 }}>Hereditary Precocious Puberty Atlas</h1>
        <div style={{ opacity: 0.85, marginTop: 6, fontSize: 13 }}>
          Complete 8-Gene Reference · MKRN3 · DLK1 · KISS1 · KISS1R · LIN28B · GNAS · LEPR · GNRH1
        </div>
        <div style={{ marginTop: 10, display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12 }}>
          <span style={{ background: 'rgba(255,255,255,0.18)', borderRadius: 6, padding: '4px 10px' }}>
            320 Patients · 8 × 40 · Seeds 2494–2501
          </span>
          <span style={{ background: 'rgba(255,255,255,0.18)', borderRadius: 6, padding: '4px 10px' }}>
            Central CPP (GnRHa-Responsive) + Peripheral MAS (Aromatase Inhibitor)
          </span>
          <span style={{ background: 'rgba(255,255,255,0.18)', borderRadius: 6, padding: '4px 10px' }}>
            Imprinting · Kisspeptin · RNA Biology · Somatic Mosaic · Metabolic
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24, borderBottom: '2px solid #e8e0f0', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              padding: '10px 20px', border: 'none', borderRadius: '8px 8px 0 0',
              background: tab === t ? accentColor : '#f3edf7',
              color: tab === t ? '#fff' : '#555', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
              borderBottom: tab === t ? `3px solid ${accentColor}` : '3px solid transparent',
              fontSize: 14,
            }}>{t}</button>
        ))}
      </div>

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 16, marginBottom: 24 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Genes Covered', value: overview.genes_covered },
              { label: 'Cohort / Gene', value: 40 },
              { label: 'Seed Range', value: overview.seed_range },
            ].map(m => (
              <div key={m.label} style={{ background: '#f3edf7', borderRadius: 10, padding: '16px 20px', textAlign: 'center', border: `1px solid #d1c4e9` }}>
                <div style={{ fontSize: 28, fontWeight: 800, color: accentColor }}>{m.value}</div>
                <div style={{ fontSize: 12, color: '#666', marginTop: 4 }}>{m.label}</div>
              </div>
            ))}
          </div>

          {/* Emergency rules */}
          <div style={{ background: '#fce4ec', borderRadius: 10, padding: 20, marginBottom: 20, border: '1px solid #f48fb1' }}>
            <div style={{ fontWeight: 700, color: '#880e4f', marginBottom: 10, fontSize: 15 }}>Emergency / Key Rules</div>
            {overview.key_emergency_rules?.map((r, i) => (
              <div key={i} style={{ padding: '5px 0', borderBottom: '1px solid #f8bbd0', fontSize: 13, color: '#4a0e2b' }}>
                ⚠ {r}
              </div>
            ))}
          </div>

          {/* Cohort cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14 }}>
            {overview.cohort_breakdown?.map(c => (
              <div key={c.gene}
                style={{ borderRadius: 10, border: `2px solid ${GENE_COLORS[c.gene] || '#ccc'}`, background: '#fff', padding: 14, cursor: 'pointer' }}
                onClick={() => { setSelectedGene(c.gene); setTab('Clinical Atlas'); }}>
                <div style={{ fontWeight: 800, fontSize: 18, color: GENE_COLORS[c.gene] || '#333' }}>{c.gene}</div>
                <div style={{ fontSize: 11, color: '#666', marginBottom: 6 }}>{c.locus} · {c.protein_size}</div>
                <div style={{ fontSize: 11, color: '#888', lineHeight: 1.4 }}>{c.disease_summary?.slice(0, 90)}…</div>
                <div style={{ marginTop: 8, fontSize: 11, color: '#999' }}>
                  {c.patients} patients · avg dx age {c.avg_age_at_dx} yr
                </div>
              </div>
            ))}
          </div>

          {/* Diagnostic tests */}
          <div style={{ marginTop: 24, background: '#e8f5e9', borderRadius: 10, padding: 20, border: '1px solid #a5d6a7' }}>
            <div style={{ fontWeight: 700, color: '#1b5e20', marginBottom: 10, fontSize: 15 }}>Key Diagnostic Tests</div>
            {overview.key_diagnostic_tests?.map((t, i) => (
              <div key={i} style={{ padding: '4px 0', borderBottom: '1px solid #c8e6c9', fontSize: 13, color: '#2e7d32' }}>
                🔬 {t}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: accentColor, color: '#fff' }}>
                {['Gene', 'Locus', 'Protein', 'Inheritance', 'Group / Disease Mechanism', 'Key Rule'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, fontSize: 12 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {genes.map((gene, idx) => {
                const bkg = idx % 2 === 0 ? '#faf5ff' : '#fff';
                const gd = breakdown?.breakdown_by_gene?.[gene];
                return (
                  <tr key={gene} style={{ background: bkg, cursor: 'pointer' }}
                    onClick={() => { setSelectedGene(gene); setTab('Clinical Atlas'); }}>
                    <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[gene], whiteSpace: 'nowrap' }}>{gene}</td>
                    <td style={{ padding: '10px 12px', color: '#555', whiteSpace: 'nowrap' }}>{gd?.locus || '—'}</td>
                    <td style={{ padding: '10px 12px', color: '#666', maxWidth: 120 }}>{gd?.protein_size || '—'}</td>
                    <td style={{ padding: '10px 12px', fontSize: 11 }}><span style={{ background: '#e8d5f5', color: accentColor, padding: '2px 6px', borderRadius: 4, whiteSpace: 'nowrap' }}>{INHERITANCE[gene]}</span></td>
                    <td style={{ padding: '10px 12px', color: '#555', maxWidth: 200, fontSize: 12 }}>{CPP_GROUP[gene]}</td>
                    <td style={{ padding: '10px 12px', color: '#333', maxWidth: 220, fontSize: 12 }}>{GENE_DISEASE[gene]?.slice(0, 100)}…</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && (
        <div style={{ display: 'flex', gap: 20 }}>
          {/* Gene selector */}
          <div style={{ width: 160, flexShrink: 0 }}>
            {genes.map(gene => (
              <button key={gene} onClick={() => setSelectedGene(gene)}
                style={{
                  display: 'block', width: '100%', marginBottom: 6, padding: '8px 12px',
                  background: selectedGene === gene ? GENE_COLORS[gene] : '#f3edf7',
                  color: selectedGene === gene ? '#fff' : GENE_COLORS[gene],
                  border: `2px solid ${GENE_COLORS[gene]}`, borderRadius: 8,
                  cursor: 'pointer', fontWeight: selectedGene === gene ? 700 : 500, fontSize: 13, textAlign: 'left',
                }}>{gene}</button>
            ))}
          </div>

          {/* Gene detail */}
          {gd && (
            <div style={{ flex: 1 }}>
              <div style={{ borderRadius: 10, background: GENE_COLORS[selectedGene], color: '#fff', padding: '16px 20px', marginBottom: 16 }}>
                <div style={{ fontSize: 22, fontWeight: 800 }}>{selectedGene}</div>
                <div style={{ fontSize: 12, opacity: 0.85, marginTop: 4 }}>{gd.locus} · {gd.protein_size} · {INHERITANCE[selectedGene]}</div>
                <div style={{ fontSize: 13, marginTop: 8, opacity: 0.9 }}>{gd.disease_category}</div>
              </div>

              {[
                { label: 'Inheritance & Mechanism', text: gd.inheritance },
                { label: 'Disease Pathway', text: gd.disease_pathway },
                { label: 'Pathognomonic Pearls', text: gd.pathognomonic },
                { label: 'Treatment', text: gd.treatment },
                { label: 'DDx', text: gd.key_ddx },
                { label: 'Cascade Testing', text: gd.cascade_testing },
                { label: 'Emergency Protocol', text: gd.emergency_protocol },
              ].map(s => s.text && (
                <div key={s.label} style={{ marginBottom: 12, background: '#faf5ff', borderRadius: 8, padding: '12px 16px', border: `1px solid #d1c4e9` }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[selectedGene], marginBottom: 6, fontSize: 13 }}>{s.label}</div>
                  <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6, whiteSpace: 'pre-wrap' }}>{s.text}</div>
                </div>
              ))}

              {/* Key features */}
              {gd.key_features?.length > 0 && (
                <div style={{ background: '#e8eaf6', borderRadius: 8, padding: '12px 16px', marginBottom: 12 }}>
                  <div style={{ fontWeight: 700, color: '#1a237e', marginBottom: 8, fontSize: 13 }}>Key Features</div>
                  {gd.key_features.map((f, i) => (
                    <div key={i} style={{ padding: '3px 0', fontSize: 13, color: '#283593' }}>• {f}</div>
                  ))}
                </div>
              )}

              {/* Patient distribution */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 12 }}>
                {[
                  { label: 'Presentations', data: gd.presentation_distribution },
                  { label: 'Managements', data: gd.management_distribution },
                  { label: 'Outcomes', data: gd.outcome_distribution },
                ].map(({ label, data }) => data && (
                  <div key={label} style={{ background: '#fff', border: '1px solid #d1c4e9', borderRadius: 8, padding: 12 }}>
                    <div style={{ fontWeight: 700, fontSize: 12, color: GENE_COLORS[selectedGene], marginBottom: 8 }}>{label}</div>
                    {Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k, v]) => (
                      <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, padding: '2px 0', borderBottom: '1px solid #ede7f6' }}>
                        <span style={{ color: '#555', maxWidth: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{k.replaceAll('_', ' ')}</span>
                        <span style={{ fontWeight: 700, color: GENE_COLORS[selectedGene] }}>{v}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ background: '#f3edf7', borderRadius: 10, padding: '12px 18px', marginBottom: 20, border: `1px solid #d1c4e9` }}>
            <div style={{ fontWeight: 700, color: accentColor, fontSize: 15 }}>{definitions.atlas_domain}</div>
            <div style={{ marginTop: 6, fontSize: 12, color: '#666' }}>
              Genes: {definitions.genes_in_atlas?.join(' · ')} · {definitions.total_patients_modelled} patients modelled
            </div>
          </div>

          {/* Drug contraindications */}
          {definitions.key_drug_contraindications?.length > 0 && (
            <div style={{ background: '#fce4ec', borderRadius: 10, padding: 18, marginBottom: 20, border: '1px solid #f48fb1' }}>
              <div style={{ fontWeight: 700, color: '#880e4f', marginBottom: 10, fontSize: 14 }}>Key Drug Contraindications & Rules</div>
              {definitions.key_drug_contraindications.map((d, i) => (
                <div key={i} style={{ padding: '5px 0', borderBottom: '1px solid #f8bbd0', fontSize: 13, color: '#4a0e2b' }}>
                  🚫 {d}
                </div>
              ))}
            </div>
          )}

          {/* Key definitions */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {definitions.key_definitions && Object.entries(definitions.key_definitions).map(([key, val]) => (
              <div key={key} style={{ background: '#faf5ff', borderRadius: 8, padding: '14px 16px', border: '1px solid #d1c4e9' }}>
                <div style={{ fontWeight: 700, color: accentColor, marginBottom: 8, fontSize: 13 }}>
                  {key.replaceAll('_', ' ')}
                </div>
                <div style={{ fontSize: 12, color: '#333', lineHeight: 1.6 }}>{val}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
