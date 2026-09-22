'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cns-brain-tumor-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'NF1':      '#1b5e20',  // deep green    -- optic glioma PATHOGNOMONIC; MPNST 8-13%; selumetinib FDA2020
  'NF2':      '#0d47a1',  // deep blue     -- bilateral VS PATHOGNOMONIC; meningioma 50-80%; bevacizumab
  'VHL':      '#4a148c',  // deep purple   -- hemangioblastoma PATHOGNOMONIC; ccRCC; belzutifan FDA2021
  'TSC1':     '#e65100',  // deep orange   -- SEGA PATHOGNOMONIC; cortical tubers; everolimus FDA
  'PTEN':     '#880e4f',  // deep pink     -- Lhermitte-Duclos PATHOGNOMONIC; macrocephaly; breast 85%
  'SMARCB1':  '#37474f',  // dark slate    -- ATRT under-3yr PATHOGNOMONIC; INI1-IHC-loss; tazemetostat
  'SUFU':     '#bf360c',  // burnt orange  -- SHH medulloblastoma HIGHEST germline; vismodegib CI children
  'TP53':     '#b71c1c',  // deep red      -- LFS; DIPG PATHOGNOMONIC; AVOID RADIATION ABSOLUTELY
};

const GENE_INFO = {
  'NF1':     { full: 'Optic-Glioma-Grade-II-PATHOGNOMONIC-15-20pct / MPNST-8-13pct-HIGHEST-Sarcoma-NF1 / Café-au-Lait-Macules-PATHOGNOMONIC / Lisch-Nodules-PATHOGNOMONIC / Selumetinib-FDA2020-Optic-Glioma / AVOID-Radiation-Secondary-MPNST-Risk',  locus: '17q11.2', size: '2839 aa / 319 kDa', inh: 'AD LOF' },
  'NF2':     { full: 'Bilateral-VS-PATHOGNOMONIC-90-95pct-NF2 / Meningioma-50-80pct / Ependymoma-Spinal-30-53pct / Bevacizumab-Hearing-Preservation / Annual-Audiogram-MANDATORY / Juvenile-Posterior-Subcapsular-Cataract-80pct-PATHOGNOMONIC', locus: '22q12.2', size: '595 aa / 66 kDa',   inh: 'AD LOF' },
  'VHL':     { full: 'Hemangioblastoma-Cerebellar-Spinal-PATHOGNOMONIC / Retinal-Hemangioblastoma-PATHOGNOMONIC / ccRCC-24-45pct-50x / Belzutifan-HIF2alpha-FDA2021 / ELST-Endolymphatic-Sac-PATHOGNOMONIC / VHL-Type1-No-Pheo-vs-Type2-Pheo',  locus: '3p25.3',  size: '213 aa / 24 kDa',   inh: 'AD LOF' },
  'TSC1':    { full: 'SEGA-5-20pct-PATHOGNOMONIC / Cortical-Tubers-PATHOGNOMONIC / Everolimus-mTOR-FDA-SEGA / Cardiac-Rhabdomyoma-Neonatal-PATHOGNOMONIC / Shagreen-Patch-PATHOGNOMONIC / Epilepsy-85pct / TSC1-Milder-TSC2',                   locus: '9q34.13', size: '1164 aa / 130 kDa', inh: 'AD LOF' },
  'PTEN':    { full: 'Lhermitte-Duclos-PATHOGNOMONIC-Adult-PTEN-Until-Proven / Macrocephaly-90pct-PATHOGNOMONIC / Breast-85pct-HIGHEST / Thyroid-Follicular-35pct / Glioma-2-4x / Trichilemmomas-Papillomatous-Papules-PATHOGNOMONIC',          locus: '10q23.31', size: '403 aa / 47 kDa',  inh: 'AD LOF' },
  'SMARCB1': { full: 'ATRT-Under-3yr-PATHOGNOMONIC / INI1-IHC-Nuclear-Loss-PATHOGNOMONIC / Tazemetostat-EZH2i-FDA2020 / MRT-Extracranial / Schwannomatosis-NF3-Like / HSCT-Consolidation-ATRT / ICE-Intrathecal-Multi-Agent',                  locus: '22q11.23', size: '385 aa / 44 kDa',  inh: 'AD LOF' },
  'SUFU':    { full: 'SHH-Medulloblastoma-HIGHEST-Germline-50-60pct / Desmoplastic-Nodular-MB-PATHOGNOMONIC / Infant-Onset-Median-2yr / Vismodegib-AVOID-Children-Permanent-Growth-Plate / AVOID-Radiation-Young-Children / PTCH1-vs-SUFU-BCC-vs-MB', locus: '10q24.32', size: '484 aa / 54 kDa',  inh: 'AD LOF' },
  'TP53':    { full: 'DIPG-H3K27M-PATHOGNOMONIC-LFS-20-30pct / AVOID-RADIATION-ABSOLUTELY / WBMRI-Toronto-Annual-NOT-CT / GBM-2-5pct / Sarcoma-25-30pct-DOMINANT / ONC201-Dordaviprone-FDA2022-H3K27M / R337H-Brazilian-Founder-1in300',        locus: '17p13.1',  size: '393 aa / 43 kDa',  inh: 'AD LOF' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#1565c0', color: '#fff', borderRadius: 4,
      padding: '2px 8px', fontSize: 11, fontWeight: 700, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneBar({ gene, pct, color, label }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 2 }}>
        <span style={{ fontWeight: 700, color }}>{gene}{label ? ` — ${label}` : ''}</span>
        <span style={{ color: '#333' }}>{pct}%</span>
      </div>
      <div style={{ background: '#e0e0e0', borderRadius: 4, height: 14 }}>
        <div style={{ background: color, width: `${Math.min(pct, 100)}%`, height: '100%', borderRadius: 4, transition: 'width 0.6s ease' }} />
      </div>
    </div>
  );
}

export default function HereditoryCNSBrainTumorAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#1b5e20' }}>Loading Hereditary CNS/Brain Tumor Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#b71c1c' }}>Error: {error}</div>;

  const ov = overview;
  const genes = ov?.genes || [];

  return (
    <div style={{ fontFamily: 'system-ui,sans-serif', background: '#f9f9f9', minHeight: '100vh', padding: 0 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1b5e20 0%,#2e7d32 100%)', color: '#fff', padding: '28px 32px 20px' }}>
        <div style={{ fontSize: 11, opacity: 0.8, letterSpacing: 1, textTransform: 'uppercase' }}>Hereditary Cancer Predisposition Atlas</div>
        <h1 style={{ margin: '6px 0 4px', fontSize: 26, fontWeight: 800 }}>Hereditary CNS / Brain Tumor Predisposition Atlas</h1>
        <div style={{ fontSize: 13, opacity: 0.9 }}>
          Complete 8-Gene Reference — NF1 · NF2 · VHL · TSC1 · PTEN · SMARCB1 · SUFU · TP53
        </div>
        <div style={{ fontSize: 12, opacity: 0.75, marginTop: 4 }}>
          {ov?.total_patients} patients · seeds {ov?.seed_range} · mean age {ov?.mean_age_dx}yr
        </div>
      </div>

      {/* Stats bar */}
      <div style={{ background: '#fff', borderBottom: '1px solid #e0e0e0', padding: '12px 32px', display: 'flex', gap: 28, flexWrap: 'wrap' }}>
        {[
          { label: 'CR Rate',         val: `${ov?.cr_pct}%`           },
          { label: 'GTR Resection',   val: `${ov?.gtr_pct}%`          },
          { label: 'Targeted Therapy',val: `${ov?.targeted_pct}%`     },
          { label: 'Radiation Used',  val: `${ov?.radiation_pct}%`    },
          { label: 'Relapse Rate',    val: `${ov?.relapse_pct}%`      },
          { label: 'Genes',           val: genes.length               },
          { label: 'Total Patients',  val: ov?.total_patients         },
        ].map(s => (
          <div key={s.label} style={{ textAlign: 'center' }}>
            <div style={{ fontSize: 20, fontWeight: 800, color: '#1b5e20' }}>{s.val}</div>
            <div style={{ fontSize: 11, color: '#666' }}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <div style={{ borderBottom: '2px solid #e0e0e0', background: '#fff', padding: '0 32px', display: 'flex', gap: 0 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '12px 20px', border: 'none', background: 'none', cursor: 'pointer',
            fontWeight: tab === t ? 700 : 400, fontSize: 14,
            borderBottom: tab === t ? '3px solid #1b5e20' : '3px solid transparent',
            color: tab === t ? '#1b5e20' : '#555',
          }}>{t}</button>
        ))}
      </div>

      <div style={{ padding: '24px 32px' }}>

        {/* OVERVIEW TAB */}
        {tab === 'Overview' && ov && (
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24, marginBottom: 24 }}>
              {/* CR by gene */}
              <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
                <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#1b5e20' }}>CR Rate by Gene</h3>
                {(ov.gene_summaries || []).map(g => (
                  <GeneBar key={g.gene} gene={g.gene} pct={g.cr_pct} color={GENE_COLORS[g.gene] || '#388e3c'} label={`n=${g.n_patients}`} />
                ))}
              </div>
              {/* Top tumor types */}
              <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
                <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#1b5e20' }}>Top Tumor Types</h3>
                {Object.entries(ov.top_tumor_types || {}).slice(0, 8).map(([tumor, n]) => (
                  <GeneBar key={tumor} gene={tumor.length > 36 ? tumor.slice(0, 36) + '…' : tumor} pct={Math.round(n / ov.total_patients * 100)} color='#388e3c' />
                ))}
              </div>
            </div>

            {/* Gene chips */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)', marginBottom: 24 }}>
              <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#1b5e20' }}>8-Gene Reference Panel</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                {genes.map(g => (
                  <div key={g} style={{
                    background: GENE_COLORS[g] || '#388e3c', color: '#fff', borderRadius: 8,
                    padding: '8px 14px', fontSize: 12, fontWeight: 700, cursor: 'default'
                  }}>
                    <div style={{ fontSize: 14 }}>{g}</div>
                    <div style={{ fontSize: 10, opacity: 0.85 }}>{GENE_INFO[g]?.locus}</div>
                    <div style={{ fontSize: 9, opacity: 0.7, marginTop: 2 }}>{GENE_INFO[g]?.inh}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Clinical pearls */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)', marginBottom: 24 }}>
              <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#1b5e20' }}>Clinical Pearls</h3>
              {(ov.clinical_pearls || []).map((p, i) => (
                <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8 }}>
                  <span style={{ color: '#1b5e20', fontWeight: 800, flexShrink: 0 }}>{i + 1}.</span>
                  <span style={{ fontSize: 13, color: '#333' }}>{p}</span>
                </div>
              ))}
            </div>

            {/* Management rules */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
              <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#b71c1c' }}>Critical Management Rules</h3>
              {(ov.key_management_rules || []).map((r, i) => (
                <div key={i} style={{ display: 'flex', gap: 10, marginBottom: 8 }}>
                  <span style={{ color: '#b71c1c', fontWeight: 800, flexShrink: 0 }}>!</span>
                  <span style={{ fontSize: 13, color: '#333' }}>{r}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* GENE TABLE TAB */}
        {tab === 'Gene Table' && breakdown && (
          <div>
            {(breakdown.breakdown || []).map(g => (
              <div key={g.gene} style={{
                background: '#fff', borderRadius: 8, padding: 20,
                boxShadow: '0 1px 4px rgba(0,0,0,0.08)', marginBottom: 16,
                borderLeft: `5px solid ${GENE_COLORS[g.gene] || '#388e3c'}`
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 8 }}>
                  <div>
                    <div style={{ fontSize: 18, fontWeight: 800, color: GENE_COLORS[g.gene] || '#388e3c' }}>{g.gene}</div>
                    <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{g.locus} · {GENE_INFO[g.gene]?.size}</div>
                  </div>
                  <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
                    {[
                      { label: 'n', val: g.n_patients },
                      { label: 'Mean age', val: `${g.mean_age_dx}yr` },
                      { label: 'CR', val: `${g.cr_pct}%` },
                      { label: 'GTR', val: `${g.gtr_pct}%` },
                      { label: 'Targeted', val: `${g.targeted_pct}%` },
                      { label: 'Relapse', val: `${g.relapse_pct}%` },
                    ].map(s => (
                      <div key={s.label} style={{ textAlign: 'center' }}>
                        <div style={{ fontSize: 16, fontWeight: 700, color: '#1b5e20' }}>{s.val}</div>
                        <div style={{ fontSize: 10, color: '#666' }}>{s.label}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div style={{ marginTop: 12, fontSize: 12, color: '#444' }}>
                  <strong>Cancer risk:</strong> {g.cancer_risk?.slice(0, 200)}…
                </div>
                <div style={{ marginTop: 8, fontSize: 12, color: '#880e4f' }}>
                  <strong>Pathognomonic:</strong> {g.pathognomonic?.slice(0, 200)}…
                </div>
                <div style={{ marginTop: 8 }}>
                  <strong style={{ fontSize: 12 }}>Key distinctions:</strong>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 4 }}>
                    {(g.key_distinctions || []).map((d, i) => (
                      <Badge key={i} text={d.replace(/-/g, ' ')} color={GENE_COLORS[g.gene] || '#388e3c'} />
                    ))}
                  </div>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginTop: 12 }}>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 700, color: '#666', marginBottom: 4 }}>Top Tumor Types</div>
                    {Object.entries(g.top_tumor_types || {}).map(([t, n]) => (
                      <div key={t} style={{ fontSize: 11, color: '#333' }}>{t.length > 28 ? t.slice(0, 28) + '…' : t}: {n}</div>
                    ))}
                  </div>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 700, color: '#666', marginBottom: 4 }}>Top Variants</div>
                    {Object.entries(g.top_variants || {}).map(([v, n]) => (
                      <div key={v} style={{ fontSize: 11, color: '#333', fontFamily: 'monospace' }}>{v}: {n}</div>
                    ))}
                  </div>
                  <div>
                    <div style={{ fontSize: 11, fontWeight: 700, color: '#666', marginBottom: 4 }}>Top Treatments</div>
                    {Object.entries(g.top_treatments || {}).map(([t, n]) => (
                      <div key={t} style={{ fontSize: 11, color: '#333' }}>{t.length > 28 ? t.slice(0, 28) + '…' : t}: {n}</div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* CLINICAL ATLAS TAB */}
        {tab === 'Clinical Atlas' && breakdown && (
          <div>
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)', marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 16px', fontSize: 15, color: '#1b5e20' }}>Per-Patient Sample (first 20)</h3>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#e8f5e9' }}>
                      {['Gene', 'Age Dx', 'Tumor Type', 'Resection', 'Treatment', 'Response', 'Radiation', 'Relapse', 'Variant'].map(h => (
                        <th key={h} style={{ padding: '8px 10px', textAlign: 'left', color: '#1b5e20', fontWeight: 700 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(breakdown.breakdown || []).flatMap(g => (g.patients || []).slice(0, 3)).slice(0, 24).map((p, i) => (
                      <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f5f5f5', borderBottom: '1px solid #eee' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[p.gene] || '#388e3c' }}>{p.gene}</td>
                        <td style={{ padding: '6px 10px' }}>{p.age_dx}yr</td>
                        <td style={{ padding: '6px 10px' }}>{(p.tumor_type || '').length > 30 ? p.tumor_type.slice(0, 30) + '…' : p.tumor_type}</td>
                        <td style={{ padding: '6px 10px' }}>{p.resection}</td>
                        <td style={{ padding: '6px 10px' }}>{(p.treatment || '').length > 24 ? p.treatment.slice(0, 24) + '…' : p.treatment}</td>
                        <td style={{ padding: '6px 10px', color: p.response === 'CR' ? '#2e7d32' : p.response === 'PD' ? '#c62828' : '#555', fontWeight: 700 }}>{p.response}</td>
                        <td style={{ padding: '6px 10px', color: p.radiation_received ? '#b71c1c' : '#2e7d32' }}>{p.radiation_received ? 'Yes' : 'No'}</td>
                        <td style={{ padding: '6px 10px', color: p.relapse ? '#c62828' : '#555' }}>{p.relapse ? 'Yes' : 'No'}</td>
                        <td style={{ padding: '6px 10px', fontFamily: 'monospace', fontSize: 11 }}>{p.variant}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Per-gene radiation usage */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)', marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#b71c1c' }}>Radiation Usage by Gene</h3>
              {(breakdown.breakdown || []).map(g => (
                <GeneBar key={g.gene} gene={g.gene} pct={g.radiation_pct || 0}
                  color={g.gene === 'TP53' ? '#b71c1c' : GENE_COLORS[g.gene] || '#388e3c'}
                  label={g.gene === 'TP53' ? 'AVOID RADIATION ABSOLUTELY' : `${g.radiation_n || 0}/${g.n_patients}`} />
              ))}
              <div style={{ fontSize: 12, color: '#b71c1c', marginTop: 8, fontWeight: 700 }}>
                ⚠ TP53 (LFS): 0% radiation — ABSOLUTE contraindication in all Li-Fraumeni patients
              </div>
            </div>

            {/* Targeted therapy bar */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)' }}>
              <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#1b5e20' }}>Targeted Therapy by Gene</h3>
              {(breakdown.breakdown || []).map(g => (
                <GeneBar key={g.gene} gene={g.gene} pct={g.targeted_pct || 0}
                  color={GENE_COLORS[g.gene] || '#388e3c'}
                  label={`${g.targeted_n || 0}/${g.n_patients}`} />
              ))}
            </div>
          </div>
        )}

        {/* DEFINITIONS TAB */}
        {tab === 'Definitions' && definitions && (
          <div>
            {/* Key rules */}
            <div style={{ background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)', marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 14px', fontSize: 15, color: '#b71c1c' }}>Critical Clinical Rules</h3>
              {Object.entries(definitions.key_rules || {}).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 12, padding: '10px 14px', background: '#ffebee', borderRadius: 6, borderLeft: '4px solid #b71c1c' }}>
                  <div style={{ fontWeight: 700, fontSize: 12, color: '#b71c1c', fontFamily: 'monospace', marginBottom: 4 }}>{k}</div>
                  <div style={{ fontSize: 13, color: '#333' }}>{v}</div>
                </div>
              ))}
            </div>
            {/* Per-gene definitions */}
            {Object.entries(definitions.definitions || {}).map(([gene, def]) => (
              <div key={gene} style={{
                background: '#fff', borderRadius: 8, padding: 20, boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
                marginBottom: 16, borderLeft: `5px solid ${GENE_COLORS[gene] || '#388e3c'}`
              }}>
                <div style={{ fontSize: 16, fontWeight: 800, color: GENE_COLORS[gene] || '#388e3c', marginBottom: 6 }}>
                  {gene} — {def.locus}
                </div>
                <div style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>{def.inheritance?.slice(0, 250)}</div>
                <div style={{ fontSize: 12, color: '#333', marginBottom: 8 }}>
                  <strong>Cancer risk:</strong> {def.cancer_risk?.slice(0, 300)}
                </div>
                <div style={{ fontSize: 12, color: '#880e4f', marginBottom: 8 }}>
                  <strong>Pathognomonic:</strong> {def.pathognomonic?.slice(0, 250)}
                </div>
                <div style={{ fontSize: 12, color: '#1b5e20', marginBottom: 8 }}>
                  <strong>Surveillance key:</strong> {def.surveillance_key?.slice(0, 250)}
                </div>
                <div>
                  {(def.key_distinctions || []).map((d, i) => (
                    <Badge key={i} text={d.replace(/-/g, ' ')} color={GENE_COLORS[gene] || '#388e3c'} />
                  ))}
                </div>
              </div>
            ))}
            {/* Cascade testing */}
            <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 16, marginTop: 8 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#1b5e20', marginBottom: 6 }}>Cascade Testing Rule</div>
              <div style={{ fontSize: 12, color: '#333' }}>{definitions.cascade_testing_rule}</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
