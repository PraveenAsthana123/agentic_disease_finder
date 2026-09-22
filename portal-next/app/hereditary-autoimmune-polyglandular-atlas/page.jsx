'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-autoimmune-polyglandular-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'AIRE':  '#b71c1c',  // deep red      — APS-1/APECED, CMC→HP→Addison
  'FOXP3': '#e65100',  // deep orange   — IPEX, neonatal T1DM, absent Tregs
  'CTLA4': '#1565c0',  // deep blue     — CHAI, lymphoproliferation, abatacept
  'LRBA':  '#4a148c',  // deep purple   — LRBA deficiency, CTLA-4 recycling defect
  'IL2RA': '#00695c',  // dark teal     — CD25 deficiency, elevated IL-2
  'STAT3': '#f9a825',  // amber         — STAT3 GOF, T1DM+thyroiditis+short stature
  'STAT1': '#2e7d32',  // dark green    — STAT1 GOF, CMC+aneurysm
  'ITCH':  '#6a1b9a',  // purple        — ITCH deficiency, syndromic dysmorphic
};

const GENE_INFO = {
  'AIRE': {
    full: 'AIRE / Autoimmune Regulator / 545aa',
    locus: '21q22.3',
    size: '545 aa / 58 kDa (SAND+PHD1+PHD2+CARD domains; mTEC thymic transcription factor; ectopic TRA expression → negative selection; LOF → autoreactive T-cells escape → APS-1; anti-IFN-ω pathognomonic; AR biallelic)',
    inh: 'AR',
  },
  'FOXP3': {
    full: 'FOXP3 / Forkhead Box P3 / 431aa',
    locus: 'Xp11.23',
    size: '431 aa / 47 kDa (master Treg transcription factor; forkhead domain; LOF → absent Tregs → IPEX; neonatal T1DM+diarrhoea+eczema; HSCT curative; XL: males affected; rapamycin preferred bridge)',
    inh: 'XL',
  },
  'CTLA4': {
    full: 'CTLA4 / Cytotoxic T-Lymphocyte Antigen 4 / 223aa',
    locus: '2q33.2',
    size: '223 aa / 25 kDa (Ig superfamily; binds CD80/CD86; transendocytosis removes APC ligands; haploinsufficiency → impaired Treg suppression; AD; abatacept replaces deficient CTLA-4; thyroiditis/cytopenias/lymphoproliferation; adult onset)',
    inh: 'AD',
  },
  'LRBA': {
    full: 'LRBA / LPS-Responsive Beige-Like Anchor Protein / 2863aa',
    locus: '4q31.3',
    size: '2863 aa / 319 kDa (BEACH+WD40+ARM domains; endosomal CTLA-4 recycling; LOF → CTLA-4 lysosomal degradation → functional CTLA-4 deficiency; CVID + autoimmunity + lymphoproliferation; AR; abatacept curative; CTLA-4 absent on Treg flow)',
    inh: 'AR',
  },
  'IL2RA': {
    full: 'IL2RA / Interleukin-2 Receptor Alpha / CD25 / 272aa',
    locus: '10p15.1',
    size: '272 aa / 30 kDa (high-affinity IL-2Rα chain; CD25; Treg constitutive expression; IL-2 consumption/sink for Tregs; LOF → Treg IL-2 starvation → autoimmunity; SERUM IL-2 VERY HIGH; AR: females also affected; IPEX-like)',
    inh: 'AR',
  },
  'STAT3': {
    full: 'STAT3 / Signal Transducer and Activator of Transcription 3 / 770aa',
    locus: '17q21.2',
    size: '770 aa / 92 kDa (JAK-STAT3 downstream; SH2+coiled-coil+DBD; GOF → excess pSTAT3 → Th17 excess + Treg impairment; T1DM+thyroiditis+short stature+lymphoproliferation; AD GOF; JAK inhibitors effective; OPPOSE LOF = Hyper-IgE)',
    inh: 'AD GOF',
  },
  'STAT1': {
    full: 'STAT1 / Signal Transducer and Activator of Transcription 1 / 750aa',
    locus: '2q32.2',
    size: '750 aa / 84 kDa (IFN-α/β/γ downstream; coiled-coil+DBD+SH2; GOF → excess IFN signalling → suppressed Th17 → CMC; thyroiditis T1DM; INTRACRANIAL ANEURYSM 15% (MRA mandatory); AD GOF; ruxolitinib effective)',
    inh: 'AD GOF',
  },
  'ITCH': {
    full: 'ITCH / ITCH E3 Ubiquitin Protein Ligase / 864aa',
    locus: '20q11.22',
    size: '864 aa / 97 kDa (HECT E3 ligase; 4 WW domains; ubiquitylates JunB→Th2 cytokine control; CTLA-4 trafficking; LOF → JunB stable → IL-4/5 excess + Treg dysfunction; UNIQUE: dysmorphic features + ID + autoimmunity; AR; rarest APS gene)',
    inh: 'AR',
  },
};

const SYNDROME_COLORS = {
  'APS1-APECED': '#b71c1c',
  'IPEX-Tregs-Absent': '#e65100',
  'CHAI-Lymphoproliferation': '#1565c0',
  'LRBA-CVID-Autoimmunity': '#4a148c',
  'CD25-IL2-Excess': '#00695c',
  'STAT3-GOF-T1DM': '#f9a825',
  'STAT1-GOF-CMC-Aneurysm': '#2e7d32',
  'ITCH-Syndromic-DD': '#6a1b9a',
};

export default function HereditaryAutoimmunePGAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function load(t) {
    setLoading(true); setError(null);
    try {
      if (t === 'Overview' && !overview) {
        const r = await fetch(`${API}/api/${SLUG}/overview`);
        setOverview(await r.json());
      } else if ((t === 'Gene Table' || t === 'Clinical Atlas') && !breakdown) {
        const r = await fetch(`${API}/api/${SLUG}/breakdown`);
        setBreakdown(await r.json());
      } else if (t === 'Definitions' && !definitions) {
        const r = await fetch(`${API}/api/${SLUG}/definitions`);
        setDefinitions(await r.json());
      }
    } catch (e) { setError(String(e)); }
    setLoading(false);
  }

  useEffect(() => { load('Overview'); }, []);
  useEffect(() => { load(tab); }, [tab]);

  return (
    <div style={{ background: '#0f172a', minHeight: '100vh', color: '#e2e8f0', fontFamily: 'monospace', padding: 24 }}>
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 22, fontWeight: 800, color: '#fbbf24', marginBottom: 4 }}>
          🧬 Hereditary Autoimmune Polyglandular Syndrome Atlas
        </div>
        <div style={{ fontSize: 12, color: '#94a3b8' }}>
          Complete 8-Gene APS / Polyendocrine Autoimmunity Reference — AIRE · FOXP3 · CTLA4 · LRBA · IL2RA · STAT3 · STAT1 · ITCH — 320 patients, seeds 2966–2973
        </div>
      </div>

      {/* ── TAB BAR ── */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 12, fontWeight: 600,
            background: tab === t ? '#fbbf24' : '#1e293b', color: tab === t ? '#0f172a' : '#94a3b8',
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#fbbf24' }}>Loading…</div>}
      {error   && <div style={{ color: '#ef4444' }}>Error: {error}</div>}

      {/* ── OVERVIEW ── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Genes', value: overview.total_genes },
              { label: 'Total Patients', value: overview.total_patients },
              { label: 'Seed Range', value: overview.seed_range },
              { label: 'AR LOF Genes', value: 5 },
              { label: 'AD Genes', value: 2 },
              { label: 'XL Genes', value: 1 },
            ].map(kpi => (
              <div key={kpi.label} style={{ background: '#1e293b', borderRadius: 8, padding: 14, textAlign: 'center' }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: '#fbbf24' }}>{kpi.value}</div>
                <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{kpi.label}</div>
              </div>
            ))}
          </div>

          {/* Gene chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
            {(overview.genes || []).map(g => (
              <div key={g} style={{ background: GENE_COLORS[g] || '#334155', borderRadius: 6, padding: '4px 12px', fontSize: 11, fontWeight: 700, color: '#fff' }}>
                {g} — {GENE_INFO[g]?.locus || '?'} — {GENE_INFO[g]?.inh || '?'}
              </div>
            ))}
          </div>

          {/* Key clinical rules */}
          <div style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <div style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 10, fontSize: 13 }}>Key Clinical Rules</div>
            {(overview.key_clinical_rules || []).map((r, i) => (
              <div key={i} style={{ fontSize: 11, color: '#cbd5e1', marginBottom: 6, paddingLeft: 12, borderLeft: '3px solid #fbbf24', lineHeight: 1.6 }}>
                {r}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── GENE TABLE ── */}
      {tab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
            <thead>
              <tr style={{ background: '#1e293b' }}>
                {['Gene', 'Locus', 'Inh', 'N', 'Mean Age Dx', 'Treg%', 'Anti-IFNω%', 'Candida%', 'HP%', 'Addison%', 'T1DM%', 'Top Treatment'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', color: '#fbbf24', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(breakdown.genes || []).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#0f172a' : '#1e293b' }}>
                  <td style={{ padding: '6px 10px', color: GENE_COLORS[g.gene] || '#e2e8f0', fontWeight: 700 }}>{g.gene}</td>
                  <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.locus}</td>
                  <td style={{ padding: '6px 10px', color: '#cbd5e1' }}>{GENE_INFO[g.gene]?.inh || '?'}</td>
                  <td style={{ padding: '6px 10px', color: '#e2e8f0' }}>{g.n_patients}</td>
                  <td style={{ padding: '6px 10px', color: '#e2e8f0' }}>{g.mean_age_dx}</td>
                  <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{g.mean_treg_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.anti_ifnw_pct > 50 ? '#fbbf24' : '#94a3b8' }}>{g.anti_ifnw_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.candida_pct > 50 ? '#f97316' : '#94a3b8' }}>{g.candida_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.hp_pct > 40 ? '#a78bfa' : '#94a3b8' }}>{g.hp_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.addison_pct > 40 ? '#ef4444' : '#94a3b8' }}>{g.addison_pct}%</td>
                  <td style={{ padding: '6px 10px', color: g.t1dm_pct > 40 ? '#22d3ee' : '#94a3b8' }}>{g.t1dm_pct}%</td>
                  <td style={{ padding: '6px 10px', color: '#64748b', fontSize: 10 }}>
                    {Object.entries(g.treatment_breakdown || {}).sort((a,b) => b[1]-a[1])[0]?.[0] || '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── CLINICAL ATLAS ── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {(breakdown.genes || []).map((g, i) => (
            <div key={g.gene} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 16, borderLeft: `4px solid ${GENE_COLORS[g.gene] || '#475569'}` }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 10 }}>
                <span style={{ background: GENE_COLORS[g.gene] || '#334155', color: '#fff', fontWeight: 800, borderRadius: 6, padding: '3px 10px', fontSize: 13 }}>{g.gene}</span>
                <span style={{ color: '#94a3b8', fontSize: 11 }}>{g.locus}</span>
                <span style={{ color: '#64748b', fontSize: 11 }}>{GENE_INFO[g.gene]?.inh || '?'}</span>
                <span style={{ color: '#475569', fontSize: 11 }}>n={g.n_patients}</span>
              </div>

              {/* KPI strip */}
              <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
                {[
                  { label: 'Age Dx', value: g.mean_age_dx + 'y', color: '#fbbf24' },
                  { label: 'Treg%', value: g.mean_treg_pct + '%', color: '#94a3b8' },
                  { label: 'Anti-IFNω', value: g.anti_ifnw_pct + '%', color: '#fbbf24' },
                  { label: 'Candida', value: g.candida_pct + '%', color: '#f97316' },
                  { label: 'HP', value: g.hp_pct + '%', color: '#a78bfa' },
                  { label: 'Addison', value: g.addison_pct + '%', color: '#ef4444' },
                  { label: 'T1DM', value: g.t1dm_pct + '%', color: '#22d3ee' },
                ].map(kpi => (
                  <div key={kpi.label} style={{ background: '#0f172a', borderRadius: 6, padding: '4px 10px', fontSize: 10 }}>
                    <span style={{ color: '#64748b' }}>{kpi.label}: </span>
                    <span style={{ color: kpi.color, fontWeight: 700 }}>{kpi.value}</span>
                  </div>
                ))}
              </div>

              {/* Inheritance summary */}
              <div style={{ fontSize: 11, color: '#cbd5e1', whiteSpace: 'pre-line', lineHeight: 1.7, marginBottom: 8, maxHeight: 220, overflowY: 'auto' }}>
                {g.inheritance}
              </div>

              {/* Patient micro-table (first 8) */}
              {g.patients && g.patients.length > 0 && (
                <div style={{ overflowX: 'auto', marginTop: 8 }}>
                  <table style={{ fontSize: 10, borderCollapse: 'collapse', width: '100%' }}>
                    <thead>
                      <tr style={{ background: '#0f172a' }}>
                        {['ID', 'Age Dx', 'Treg%', 'Anti-IFNω', 'Candida', 'HP', 'Addison', 'T1DM', 'Treatment'].map(h => (
                          <th key={h} style={{ padding: '3px 8px', color: '#64748b', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {g.patients.slice(0, 8).map(p => (
                        <tr key={p.id}>
                          <td style={{ padding: '3px 8px', color: '#475569' }}>{p.id}</td>
                          <td style={{ padding: '3px 8px', color: '#e2e8f0' }}>{p.age_at_diagnosis}</td>
                          <td style={{ padding: '3px 8px', color: '#94a3b8' }}>{p.treg_pct}</td>
                          <td style={{ padding: '3px 8px', color: p.anti_ifnw_positive ? '#fbbf24' : '#475569' }}>{p.anti_ifnw_positive ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: p.candida_infection ? '#f97316' : '#475569' }}>{p.candida_infection ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: p.hypoparathyroidism ? '#a78bfa' : '#475569' }}>{p.hypoparathyroidism ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: p.addison_disease ? '#ef4444' : '#475569' }}>{p.addison_disease ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: p.type1_diabetes ? '#22d3ee' : '#475569' }}>{p.type1_diabetes ? 'Yes' : 'No'}</td>
                          <td style={{ padding: '3px 8px', color: '#94a3b8', fontSize: 10 }}>{p.treatment}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'Definitions' && definitions && (
        <div>
          <div style={{ marginBottom: 12, color: '#64748b', fontSize: 12 }}>{definitions.count} clinical definitions</div>
          {(definitions.definitions || []).map((t, i) => (
            <div key={i} style={{ background: '#1e293b', borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, color: '#fbbf24', marginBottom: 8, fontSize: 13 }}>{t.term}</div>
              <div style={{ display: 'flex', gap: 6, marginBottom: 8, flexWrap: 'wrap' }}>
                {(t.genes || []).map(g => (
                  <span key={g} style={{ background: GENE_COLORS[g] || '#334155', color: '#fff', fontSize: 10, borderRadius: 4, padding: '2px 8px', fontWeight: 700 }}>{g}</span>
                ))}
              </div>
              <div style={{ fontSize: 12, color: '#cbd5e1', lineHeight: 1.7, whiteSpace: 'pre-line' }}>{t.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
