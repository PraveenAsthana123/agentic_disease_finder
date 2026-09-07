'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ENG:    '#dc2626',  // red     — HHT1; pulmonary AVM; paradoxical embolism; epistaxis
  ACVRL1: '#ea580c',  // orange  — HHT2; hepatic AVM; high-output failure; GI bleeding
  SMAD4:  '#ca8a04',  // amber   — JPHT; HHT + Juvenile Polyposis; colon cancer mandatory
  KRIT1:  '#16a34a',  // green   — CCM1; most common; Q455X Mexican-American founder
  CCM2:   '#2563eb',  // blue    — CCM2; intermediate; scaffold; de novo mutations
  PDCD10: '#7c3aed',  // purple  — CCM3; most aggressive; spinal CCM; meningioma
  RASA1:  '#0f766e',  // teal    — CM-AVM1; multifocal CMs; Parkes Weber; brain AVM
  TEK:    '#be185d',  // pink    — TIE2; venous malformation; phleboliths; rapamycin
};

const GENE_DISEASE = {
  ENG:    'AD HHT1 — ENG-658aa — 9q34.11 — Pulmonary-AVM-60-80pct-Highest-Paradoxical-Embolism-Stroke-Risk — Epistaxis-UNIVERSAL-First-Decade — Bevacizumab-Anti-VEGF-Level-B — Antibiotic-Prophylaxis-MANDATORY-Brain-Abscess',
  ACVRL1: 'AD HHT2 — ACVRL1-503aa — 12q13.13 — Hepatic-AVM-Over-80pct-Dominant — High-Output-Cardiac-Failure — GI-Bleeding-Most-Prominent — Hepatic-Embolisation-ABSOLUTELY-CONTRAINDICATED',
  SMAD4:  'AD JPHT — SMAD4-552aa — 18q21.2 — ONLY-Gene-HHT-PLUS-Juvenile-Polyposis — Colon-Cancer-Risk-39-68pct — Annual-Colonoscopy-from-Age-15-Mandatory — Gastric-Cancer-Risk — Aortic-Dilation',
  KRIT1:  'AD CCM1 — KRIT1-736aa — 7q21.2 — Most-Common-Hereditary-CCM — p.Q455X-Mexican-American-Founder-40pct-CCM1-Alleles — Seizures-40-70pct — SWI-MRI-Mandatory — No-Radiosurgery',
  CCM2:   'AD CCM2 — CCM2-444aa — 7p13 — Intermediate-CCM-Phenotype — De-Novo-Mutations-20pct — Scaffold-KRIT1-PDCD10-Bridge — Panel-Testing-Not-Single-Gene — Retinal-Cavernomas',
  PDCD10: 'AD CCM3 — PDCD10-212aa — 3q26.1 — MOST-AGGRESSIVE-CCM — Youngest-Onset — Spinal-Cord-Cavernomas-DISTINCTIVE — Meningioma-Association-10pct — Annual-MRI-Not-Triennial — Miliary-Pattern',
  RASA1:  'AD CM-AVM1 — RASA1-1047aa — 5q14.3 — Multifocal-CMs-Pale-Halo-Pathognomonic — Fast-Flow-AVMs-Brain-Spine — Parkes-Weber-Limb-Hypertrophy — Brain-AVM-20pct — No-Sclerotherapy-Risk',
  TEK:    'AD+Somatic VMCM — TEK-1124aa — 9p21.2 — Most-Common-Somatic-Vascular-Malformation-Gene — Phleboliths-Pathognomonic — Rapamycin-Level-B-mTOR-Inhibition — Chronic-LIC-Coagulopathy — Pregnancy-Expansion',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#94a3b8' }}>Loading…</div>;
}

function ErrorBox({ msg }) {
  return (
    <div style={{ padding: '1rem', background: '#450a0a', borderRadius: 8, color: '#fca5a5', margin: '1rem 0' }}>
      Error: {msg}
    </div>
  );
}

function KPI({ label, value, color }) {
  return (
    <div style={{
      background: '#1e293b', borderRadius: 10, padding: '1rem 1.2rem',
      borderLeft: `4px solid ${color || '#6366f1'}`, minWidth: 150,
    }}>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#a5b4fc' }}>{value}</div>
      <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 6, padding: '2px 8px', fontSize: 11, fontWeight: 600,
    }}>{text}</span>
  );
}

export default function HereditaryVascularMalformationAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true); setError(null);
    Promise.all([
      fetch(`${API}/api/hereditary-vascular-malformation-atlas/overview`).then(r => r.json()),
      fetch(`${API}/api/hereditary-vascular-malformation-atlas/breakdown`).then(r => r.json()),
      fetch(`${API}/api/hereditary-vascular-malformation-atlas/definitions`).then(r => r.json()),
    ])
      .then(([ov, bd, df]) => { setOverview(ov); setBreakdown(bd); setDefinitions(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '0.5rem 1.2rem', cursor: 'pointer', borderRadius: '6px 6px 0 0',
    fontWeight: tab === t ? 700 : 400,
    background: tab === t ? '#dc2626' : '#1e293b',
    color: tab === t ? '#fff' : '#94a3b8',
    border: 'none', fontSize: 13,
  });

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', padding: '1.5rem' }}>
      <div style={{ maxWidth: 1400, margin: '0 auto' }}>

        {/* Header */}
        <div style={{ marginBottom: '1.5rem' }}>
          <h1 style={{ fontSize: 22, fontWeight: 800, color: '#dc2626', margin: 0 }}>
            🩸 Hereditary-Vascular-Malformation-Atlas
          </h1>
          <p style={{ color: '#94a3b8', margin: '0.3rem 0 0', fontSize: 13 }}>
            Complete 8-Gene Hereditary Vascular Malformation Atlas —
            ENG · ACVRL1 · SMAD4 · KRIT1 · CCM2 · PDCD10 · RASA1 · TEK —
            320 patients · seeds 1966–1973
          </p>
          <p style={{ color: '#64748b', margin: '0.2rem 0 0', fontSize: 11 }}>
            Spectrum: HHT1/HHT2/JPHT (ENG/ACVRL1/SMAD4 — AVMs + telangiectasias) → CCM (KRIT1/CCM2/PDCD10 — cavernomas, CCM3 most aggressive) → CM-AVM (RASA1 — fast-flow + CMs) → Venous Malformation (TEK/TIE2 — rapamycin responsive)
          </p>
        </div>

        {loading && <Loading />}
        {error && <ErrorBox msg={error} />}

        {/* Tabs */}
        <div style={{ display: 'flex', gap: 4, marginBottom: 0, flexWrap: 'wrap' }}>
          {TABS.map(t => (
            <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>
          ))}
        </div>
        <div style={{ background: '#1e293b', borderRadius: '0 8px 8px 8px', padding: '1.5rem' }}>

          {/* ── OVERVIEW TAB ── */}
          {tab === 'Overview' && overview && (
            <div>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: '1.5rem' }}>
                <KPI label="Total Patients" value={overview.total_patients} color="#dc2626" />
                <KPI label="Pulmonary AVM (PAVM)" value={overview.pavm_patients} color="#ea580c" />
                <KPI label="Hepatic AVM" value={overview.hepatic_avm_patients} color="#ca8a04" />
                <KPI label="CCM Seizures" value={overview.ccm_seizure_patients} color="#16a34a" />
                <KPI label="CCM Haemorrhage" value={overview.ccm_haemorrhage_patients} color="#7c3aed" />
                <KPI label="GI Polyps (SMAD4)" value={overview.gi_polyps_patients} color="#0f766e" />
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <h3 style={{ color: '#dc2626', fontSize: 14, marginBottom: 8 }}>Pathway</h3>
                <div style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem', fontSize: 12, color: '#cbd5e1' }}>
                  {overview.pathway}
                </div>
              </div>

              <div style={{ marginBottom: '1.5rem' }}>
                <h3 style={{ color: '#dc2626', fontSize: 14, marginBottom: 8 }}>Key Clinical Insight</h3>
                <div style={{ background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem', fontSize: 12, color: '#cbd5e1' }}>
                  {overview.key_clinical_insight}
                </div>
              </div>

              <div>
                <h3 style={{ color: '#dc2626', fontSize: 14, marginBottom: 8 }}>Genes in This Atlas</h3>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {overview.genes?.map(g => (
                    <Badge key={g} text={g} color={GENE_COLORS[g] || '#6366f1'} />
                  ))}
                </div>
                <div style={{ marginTop: '1rem', display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  {overview.genes?.map(g => (
                    <div key={g} style={{
                      background: '#0f172a', borderRadius: 8, padding: '0.5rem 0.8rem',
                      borderLeft: `3px solid ${GENE_COLORS[g] || '#6366f1'}`, minWidth: 100,
                    }}>
                      <div style={{ fontWeight: 700, color: GENE_COLORS[g] || '#a5b4fc', fontSize: 13 }}>{g}</div>
                      <div style={{ color: '#94a3b8', fontSize: 10 }}>{overview.gene_patient_counts?.[g]} pts</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* ── GENE TABLE TAB ── */}
          {tab === 'Gene Table' && breakdown && (
            <div>
              <h3 style={{ color: '#dc2626', fontSize: 14, marginBottom: '1rem' }}>
                Per-Gene Breakdown — Vascular Malformation Spectrum
              </h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#0f172a', color: '#64748b' }}>
                      {['Gene', 'Locus', 'Size', 'Inh.', 'N', 'PAVM%', 'Hepatic%', 'Seizure%', 'Haem%', 'Polyp%', 'Disease / Key Fact'].map(h => (
                        <th key={h} style={{ padding: '0.5rem 0.7rem', textAlign: 'left', whiteSpace: 'nowrap', fontSize: 11 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.values(breakdown).map((g, i) => (
                      <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                        <td style={{ padding: '0.5rem 0.7rem', fontWeight: 700, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.locus}</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#94a3b8', whiteSpace: 'nowrap' }}>{g.protein_size}</td>
                        <td style={{ padding: '0.5rem 0.7rem' }}>
                          <Badge text={g.inheritance?.split(' ')[0] || g.inheritance} color={g.inheritance?.includes('AR') ? '#16a34a' : '#2563eb'} />
                        </td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#e2e8f0', fontWeight: 600 }}>{g.n_patients}</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#dc2626' }}>{g.pavm_pct}%</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#ea580c' }}>{g.hepatic_avm_pct}%</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#16a34a' }}>{g.seizure_pct}%</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#7c3aed' }}>{g.haemorrhage_pct}%</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#ca8a04' }}>{g.gi_polyps_pct}%</td>
                        <td style={{ padding: '0.5rem 0.7rem', color: '#cbd5e1', fontSize: 10, maxWidth: 280 }}>
                          {GENE_DISEASE[g.gene] || g.alt_name}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ── CLINICAL ATLAS TAB ── */}
          {tab === 'Clinical Atlas' && breakdown && (
            <div>
              <h3 style={{ color: '#dc2626', fontSize: 14, marginBottom: '1rem' }}>
                Clinical Atlas — 8 Vascular Malformation Genes
              </h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                {Object.values(breakdown).map(g => (
                  <div key={g.gene} style={{
                    background: '#0f172a', borderRadius: 10,
                    borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#6366f1'}`,
                    padding: '1rem 1.2rem',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
                      <span style={{ fontSize: 16, fontWeight: 800, color: GENE_COLORS[g.gene] || '#a5b4fc' }}>{g.gene}</span>
                      <Badge text={g.inheritance?.split(' ')[0] || g.inheritance} color={g.inheritance?.includes('AD') ? '#2563eb' : '#16a34a'} />
                      <span style={{ color: '#64748b', fontSize: 11 }}>{g.locus} · {g.protein_size}</span>
                      <span style={{ color: '#64748b', fontSize: 11 }}>{g.n_patients} patients</span>
                    </div>
                    <div style={{ color: '#94a3b8', fontSize: 11, marginBottom: 8 }}>{g.alt_name}</div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 10 }}>
                      {[
                        { label: 'Age of Onset', val: g.age_of_onset, color: '#dc2626' },
                        { label: 'Key Biomarker', val: g.key_biomarker, color: '#0f766e' },
                        { label: 'Pathognomonic / DDx', val: g.pathognomonic, color: '#7c3aed' },
                        { label: 'Treatment', val: g.treatment, color: '#2563eb' },
                      ].map(({ label, val, color }) => (
                        <div key={label} style={{ background: '#1e293b', borderRadius: 8, padding: '0.7rem' }}>
                          <div style={{ fontSize: 10, color, fontWeight: 700, marginBottom: 4, textTransform: 'uppercase' }}>{label}</div>
                          <div style={{ fontSize: 11, color: '#cbd5e1', lineHeight: 1.5 }}>{val}</div>
                        </div>
                      ))}
                    </div>

                    <div style={{ marginTop: 10 }}>
                      <div style={{ fontSize: 10, color: '#ca8a04', fontWeight: 700, marginBottom: 6, textTransform: 'uppercase' }}>
                        ⚠ Critical Flags
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {g.critical_flags?.map((flag, fi) => (
                          <div key={fi} style={{
                            background: '#1e293b', borderRadius: 6, padding: '0.4rem 0.7rem',
                            borderLeft: '3px solid #ca8a04', fontSize: 11, color: '#fef3c7',
                          }}>
                            {flag}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── DEFINITIONS TAB ── */}
          {tab === 'Definitions' && definitions && (
            <div>
              <h3 style={{ color: '#dc2626', fontSize: 14, marginBottom: '1rem' }}>
                Gene Definitions &amp; Glossary
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.7rem', marginBottom: '1.5rem' }}>
                {Object.entries(definitions.gene_definitions || {}).map(([gene, def]) => (
                  <div key={gene} style={{
                    background: '#0f172a', borderRadius: 8, padding: '0.8rem 1rem',
                    borderLeft: `3px solid ${GENE_COLORS[gene] || '#6366f1'}`,
                  }}>
                    <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#a5b4fc', fontSize: 13, marginBottom: 4 }}>
                      {gene}
                    </div>
                    <div style={{ fontSize: 11, color: '#94a3b8' }}>
                      {def.locus} · {def.protein_size} · {def.inheritance}
                    </div>
                    <div style={{ fontSize: 11, color: '#cbd5e1', marginTop: 4, lineHeight: 1.4 }}>
                      {def.protein}
                    </div>
                  </div>
                ))}
              </div>

              <h3 style={{ color: '#dc2626', fontSize: 14, marginBottom: '0.8rem' }}>Glossary</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1.5rem' }}>
                {Object.entries(definitions.glossary || {}).map(([term, def]) => (
                  <div key={term} style={{
                    background: '#0f172a', borderRadius: 8, padding: '0.7rem 1rem',
                  }}>
                    <div style={{ fontWeight: 700, color: '#e2e8f0', fontSize: 12 }}>{term}</div>
                    <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 3, lineHeight: 1.4 }}>{def}</div>
                  </div>
                ))}
              </div>

              <h3 style={{ color: '#dc2626', fontSize: 14, marginBottom: '0.8rem' }}>Clinical Pearls</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                {definitions.clinical_pearls?.map((pearl, pi) => (
                  <div key={pi} style={{
                    background: '#0f172a', borderRadius: 6, padding: '0.6rem 1rem',
                    borderLeft: '3px solid #dc2626', fontSize: 11, color: '#cbd5e1',
                  }}>
                    {pearl}
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}
