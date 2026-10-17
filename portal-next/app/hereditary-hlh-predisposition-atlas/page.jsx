'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hlh-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'PRF1':   '#1a237e',  // deep navy      — FHL2 perforin null; CD107a NORMAL cytotoxicity ABSENT; emapalumab+HSCT
  'UNC13D': '#4a148c',  // deep purple    — FHL3 Munc13-4; CD107a ABSENT PATHOGNOMONIC; Middle East/Turkish
  'STX11':  '#006064',  // dark teal      — FHL4 Syntaxin-11; Kurdish; later onset; SNARE-STX11-STXBP2 complex
  'STXBP2': '#1b5e20',  // deep green     — FHL5 Munc18-2; IBD 40% HALLMARK; SNHL 30%; HSCT treats HLH+IBD
  'RAB27A': '#e65100',  // deep orange    — GS2 Griscelli Type 2; silvery hair PATHOGNOMONIC; HLH no neuro
  'LYST':   '#880e4f',  // deep magenta   — CHS Chediak-Higashi; giant granules PATHOGNOMONIC smear; HSCT not neuro
  'SH2D1A': '#bf360c',  // deep red-brown — XLP1 SAP; males only; iNKT ABSENT PATHOGNOMONIC; EBV-triggered; HSCT URGENT
  'XIAP':   '#b71c1c',  // deep red       — XLP2 BIRC4; males only; IBD 50% HALLMARK; splenomegaly; recurrent milder
};

const GENE_INFO = {
  'PRF1':   { full: 'FHL2-Perforin-Deficiency / CD107a-NORMAL-cytotox-ABSENT / Most-Common-FHL-30-40pct / Q8R-A91V-Founders / Emapalumab-FDA2018 / HSCT-Curative',                  locus: '10q22.1', size: '555 aa / 67 kDa',   inh: 'AR LOF biallelic' },
  'UNC13D': { full: 'FHL3-Munc13-4 / CD107a-ABSENT-PATHOGNOMONIC / Middle-East-Turkish-Most-Common / IVS27-Founder / Emapalumab-HSCT',                                             locus: '17q25.1', size: '2090 aa / 237 kDa', inh: 'AR LOF biallelic' },
  'STX11':  { full: 'FHL4-Syntaxin-11-SNARE / CD107a-ABSENT / Kurdish-Turkish-Rare / Later-Onset-2-6yr / STX11-STXBP2-Complex / HSCT-Curative',                                   locus: '6q24.2',  size: '287 aa / 32 kDa',   inh: 'AR LOF biallelic' },
  'STXBP2': { full: 'FHL5-Munc18-2 / IBD-40pct-HALLMARK / SNHL-30pct / CD107a-ABSENT / Neonatal-Onset / HSCT-Curative-HLH-And-IBD',                                              locus: '19p13.2', size: '592 aa / 67 kDa',   inh: 'AR LOF biallelic' },
  'RAB27A': { full: 'GS2-Griscelli-Syndrome-Type2 / Silvery-Hair-PATHOGNOMONIC / HLH-WITHOUT-Neurodegeneration / CD107a-ABSENT / HSCT-Curative-HLH-NOT-Hair',                    locus: '15q21.3', size: '221 aa / 25 kDa',   inh: 'AR LOF biallelic' },
  'LYST':   { full: 'CHS-Chediak-Higashi / Giant-Granules-PATHOGNOMONIC-Smear / Silvery-Hair / Accelerated-Phase-HLH-50-85pct / HSCT-Curative-HLH-NOT-Neurodegeneration',        locus: '1q42.3',  size: '3801 aa / 430 kDa', inh: 'AR LOF biallelic' },
  'SH2D1A': { full: 'XLP1-SAP-Deficiency / MALES-ONLY-XLR / iNKT-ABSENT-PATHOGNOMONIC / EBV-Triggered-HLH-70pct / Agammaglobulinaemia-50pct / HSCT-URGENT-MANDATORY',           locus: 'Xq25',    size: '128 aa / 15 kDa',   inh: 'XLR LOF hemizygous' },
  'XIAP':   { full: 'XLP2-BIRC4 / MALES-ONLY-XLR / IBD-50pct-HALLMARK / Splenomegaly-PROMINENT / Recurrent-Milder-HLH / iNKT-Low-Not-Absent / HSCT-Curative-HLH-IBD',          locus: 'Xq25',    size: '497 aa / 56 kDa',   inh: 'XLR LOF hemizygous' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: `${color}18`, color, border: `1px solid ${color}55`,
      borderRadius: 4, fontSize: 10, padding: '2px 7px', marginLeft: 6, fontWeight: 600,
    }}>{text}</span>
  );
}

function card(label, value, sub, color) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color}33`, borderRadius: 10,
      padding: '14px 18px', minWidth: 140, flex: 1 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 800, color }}>{value}</div>
      <div style={{ fontSize: 10, color: '#aaa' }}>{sub}</div>
    </div>
  );
}

export default function HereditaryHLHPredispositionAtlasPage() {
  const [tab, setTab]                 = useState('Overview');
  const [overview, setOverview]       = useState(null);
  const [breakdown, setBreakdown]     = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState('');
  const [expandedGene, setExpandedGene] = useState(null);
  const [expandedDef, setExpandedDef]   = useState(null);

  useEffect(() => {
    setLoading(true); setError('');
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefinitions(df); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const tabStyle = (t) => ({
    padding: '8px 18px', border: 'none', borderRadius: 6, cursor: 'pointer', fontWeight: 600,
    fontSize: 13, background: tab === t ? '#1a237e' : '#f5f5f5', color: tab === t ? '#fff' : '#555',
    marginRight: 6,
  });

  if (loading) return <div style={{ padding: 40, textAlign: 'center', color: '#888' }}>Loading Hereditary HLH Predisposition Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#c00' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'Inter,system-ui,sans-serif', background: '#f8f9fa', minHeight: '100vh', padding: 24 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1a237e 0%,#4a148c 100%)', borderRadius: 14,
        padding: '24px 32px', marginBottom: 24, color: '#fff' }}>
        <div style={{ fontSize: 11, opacity: 0.75, marginBottom: 6 }}>
          Hereditary HLH Predisposition Atlas · 8 Genes · 320 Patients · Seeds 3574–3581
        </div>
        <h1 style={{ margin: 0, fontSize: 26, fontWeight: 800 }}>
          🧬 Hereditary HLH Predisposition Atlas
        </h1>
        <div style={{ marginTop: 10, fontSize: 13, opacity: 0.88 }}>
          PRF1 · UNC13D · STX11 · STXBP2 · RAB27A · LYST · SH2D1A · XIAP
        </div>
        <div style={{ marginTop: 6, fontSize: 12, opacity: 0.75 }}>
          Familial Haemophagocytic Lymphohistiocytosis (FHL2–5) · Griscelli Syndrome Type 2 ·
          Chediak-Higashi Syndrome · XLP1 · XLP2 — Complete Germline Predisposition Reference
        </div>
      </div>

      {/* Tabs */}
      <div style={{ marginBottom: 20 }}>
        {TABS.map(t => <button key={t} style={tabStyle(t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {/* ── Overview ─────────────────────────────────────────────── */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            {card('Total Patients', overview.total_patients, '8 genes × 40 patients', '#1a237e')}
            {card('Avg Ferritin Peak', `${(overview.avg_ferritin_peak/1000).toFixed(1)}k`, 'ng/mL HLH episode', '#880e4f')}
            {card('HSCT Done', overview.hsct_done, `of ${overview.total_patients} total`, '#1b5e20')}
            {card('EBV-Triggered', overview.ebv_triggered, 'most common trigger', '#e65100')}
            {card('Avg HScore', overview.avg_hscore, '>169 = 93% sens', '#4a148c')}
            {card('Avg Age Onset', `${overview.avg_age_onset}yr`, 'all genes combined', '#006064')}
          </div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            {card('IBD Present', overview.has_ibd, 'STXBP2 40% · XIAP 50%', '#1b5e20')}
            {card('Giant Granules', overview.giant_granules, 'LYST CHS pathognomonic smear', '#880e4f')}
            {card('Silvery Hair', overview.silvery_hair, 'RAB27A GS2 · LYST CHS', '#e65100')}
            {card('HLH Recurrence', overview.recurrence, 'without HSCT', '#b71c1c')}
            {card('Agammaglobulinaemia', overview.agammaglobulinaemia, 'XLP1 50% · STXBP2 30%', '#4a148c')}
            {card('Males (XLP)', overview.males, 'SH2D1A+XIAP = males only', '#bf360c')}
          </div>

          {/* Clinical Pearl */}
          <div style={{ background: '#fff8e1', border: '2px solid #f9a825', borderRadius: 10,
            padding: '14px 18px', marginBottom: 20, fontSize: 12, color: '#5d4037' }}>
            <strong>⚡ CLINICAL PEARL:</strong> {overview.clinical_pearl}
          </div>

          {/* Gene summary cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(340px,1fr))', gap: 14 }}>
            {overview.gene_summary?.map(gs => {
              const col = GENE_COLORS[gs.gene] || '#333';
              const info = GENE_INFO[gs.gene] || {};
              return (
                <div key={gs.gene} style={{ background: '#fff', border: `2px solid ${col}33`,
                  borderRadius: 10, padding: 16 }}>
                  <div style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                    <span style={{ fontSize: 20, fontWeight: 900, color: col }}>{gs.gene}</span>
                    <Badge text={gs.inheritance} color={col} />
                    <Badge text={gs.locus} color={col} />
                  </div>
                  <div style={{ fontSize: 10, color: '#888', marginBottom: 6 }}>{info.size}</div>
                  <div style={{ fontSize: 11, color: '#555', marginBottom: 6 }}>{info.full}</div>
                  <div style={{ fontSize: 11, marginBottom: 4 }}>
                    <strong>HLH Risk:</strong> <span style={{ color: col }}>{gs.hlh_risk}</span>
                  </div>
                  <div style={{ fontSize: 11, marginBottom: 4 }}>
                    <strong>Pathognomonic:</strong> {gs.pathognomonic}
                  </div>
                  <div style={{ fontSize: 11, marginBottom: 4 }}>
                    <strong>Key Rule:</strong> {gs.key_rule}
                  </div>
                  <div style={{ fontSize: 11, color: '#c62828' }}>
                    <strong>Avoid:</strong> {gs.key_avoid}
                  </div>
                  <div style={{ marginTop: 8, fontSize: 10, color: '#666', background: `${col}09`,
                    borderRadius: 6, padding: '6px 8px' }}>
                    <strong>Targeted Rx:</strong> {gs.targeted_rx}
                  </div>
                  <div style={{ marginTop: 6, fontSize: 10, color: '#888' }}>
                    n = {gs.n} · mean age onset = {gs.mean_age_onset}yr
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Gene Table ─────────────────────────────────────────────── */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, background: '#fff',
            borderRadius: 10, overflow: 'hidden', boxShadow: '0 1px 4px #0001' }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Syndrome', 'Locus / Inh.', 'n', 'Mean Age', 'CD107a Absent', 'NK Cytotox%',
                  'Perforin%', 'iNKT%', 'Ferritin Peak', 'HScore', 'EBV Trig.', 'IBD', 'Silvery Hair',
                  'Giant Gran.', 'HSCT Done', 'Recurrence'].map(h =>
                  <th key={h} style={{ padding: '10px 8px', textAlign: 'left', fontSize: 11 }}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {Object.values(breakdown.gene_data).map((gd, i) => {
                const col = GENE_COLORS[gd.gene] || '#333';
                return (
                  <tr key={gd.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff',
                    cursor: 'pointer' }}
                    onClick={() => setExpandedGene(expandedGene === gd.gene ? null : gd.gene)}>
                    <td style={{ padding: '8px', fontWeight: 800, color: col }}>{gd.gene}</td>
                    <td style={{ padding: '8px', fontSize: 10, color: '#555', maxWidth: 160 }}>{gd.syndrome}</td>
                    <td style={{ padding: '8px', fontSize: 10 }}>{gd.locus}<br/><span style={{color:'#888'}}>{gd.inheritance}</span></td>
                    <td style={{ padding: '8px', fontWeight: 700 }}>{gd.n}</td>
                    <td style={{ padding: '8px' }}>{gd.mean_age_onset}yr</td>
                    <td style={{ padding: '8px', color: '#c62828', fontWeight: 700 }}>{gd.cd107a_absent_n}/40</td>
                    <td style={{ padding: '8px' }}>{gd.avg_nk_cytotox}%</td>
                    <td style={{ padding: '8px' }}>{gd.avg_perforin_pct}%</td>
                    <td style={{ padding: '8px' }}>{gd.avg_inkt_pct}%</td>
                    <td style={{ padding: '8px' }}>{(gd.avg_ferritin_peak/1000).toFixed(1)}k</td>
                    <td style={{ padding: '8px' }}>{gd.avg_hscore}</td>
                    <td style={{ padding: '8px' }}>{gd.ebv_triggered_n}/40</td>
                    <td style={{ padding: '8px', color: gd.ibd_n > 5 ? '#1b5e20' : '#999' }}>{gd.ibd_n}/40</td>
                    <td style={{ padding: '8px', color: gd.silvery_hair_n > 5 ? '#e65100' : '#999' }}>{gd.silvery_hair_n}/40</td>
                    <td style={{ padding: '8px', color: gd.giant_granules_n > 5 ? '#880e4f' : '#999' }}>{gd.giant_granules_n}/40</td>
                    <td style={{ padding: '8px', color: '#1b5e20', fontWeight: 700 }}>{gd.hsct_done_n}/40</td>
                    <td style={{ padding: '8px', color: '#b71c1c' }}>{gd.recurrence_n}/40</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* Expanded gene detail */}
          {expandedGene && breakdown.gene_data[expandedGene] && (() => {
            const gd  = breakdown.gene_data[expandedGene];
            const col = GENE_COLORS[expandedGene] || '#333';
            return (
              <div style={{ marginTop: 20, background: '#fff', border: `2px solid ${col}44`,
                borderRadius: 10, padding: 20 }}>
                <h3 style={{ color: col, marginTop: 0 }}>{expandedGene} — {gd.syndrome}</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
                  <div><strong>Key Rule:</strong> <span style={{ fontSize: 12 }}>{gd.key_rule}</span></div>
                  <div><strong>Key Avoid:</strong> <span style={{ fontSize: 12, color: '#c62828' }}>{gd.key_avoid}</span></div>
                  <div><strong>Pathognomonic:</strong> <span style={{ fontSize: 12 }}>{gd.pathognomonic}</span></div>
                  <div><strong>Targeted Rx:</strong> <span style={{ fontSize: 12 }}>{gd.targeted_rx}</span></div>
                </div>
                <div style={{ fontSize: 12, marginBottom: 10 }}>
                  <strong>Top 20 patients (sorted by age at HLH onset):</strong>
                </div>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr style={{ background: col, color: '#fff' }}>
                        {['Patient ID', 'Sex', 'Age (yr)', 'Trigger', 'Ferritin', 'HScore',
                          'CD107a', 'NK Cytotox%', 'Perforin%', 'iNKT%', 'IBD', 'Silvery',
                          'Giant Gran', 'HSCT', 'Outcome'].map(h =>
                          <th key={h} style={{ padding: '6px 8px', textAlign: 'left' }}>{h}</th>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {gd.patients?.map((p, ri) => (
                        <tr key={p.patient_id} style={{ background: ri % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                          <td style={{ padding: '4px 8px', fontFamily: 'monospace', fontSize: 10 }}>{p.patient_id}</td>
                          <td style={{ padding: '4px 8px' }}>{p.sex}</td>
                          <td style={{ padding: '4px 8px' }}>{p.age_at_hlh_onset}</td>
                          <td style={{ padding: '4px 8px', fontSize: 10 }}>{p.hlh_trigger}</td>
                          <td style={{ padding: '4px 8px' }}>{(p.ferritin_peak_ng_mL/1000).toFixed(1)}k</td>
                          <td style={{ padding: '4px 8px' }}>{p.hscore}</td>
                          <td style={{ padding: '4px 8px', color: p.cd107a_normal ? '#1b5e20' : '#c62828' }}>
                            {p.cd107a_normal ? `✓ ${p.cd107a_pct}%` : `✗ ${p.cd107a_pct}%`}
                          </td>
                          <td style={{ padding: '4px 8px' }}>{p.nk_cytotox_pct}%</td>
                          <td style={{ padding: '4px 8px' }}>{p.perforin_stain_pct}%</td>
                          <td style={{ padding: '4px 8px' }}>{p.inkt_pct}%</td>
                          <td style={{ padding: '4px 8px' }}>{p.has_ibd ? '✓' : '—'}</td>
                          <td style={{ padding: '4px 8px' }}>{p.silvery_hair ? '✓' : '—'}</td>
                          <td style={{ padding: '4px 8px' }}>{p.giant_granules_smear ? '✓' : '—'}</td>
                          <td style={{ padding: '4px 8px', color: p.hsct_done ? '#1b5e20' : '#888' }}>
                            {p.hsct_done ? '✓' : '—'}
                          </td>
                          <td style={{ padding: '4px 8px', fontSize: 10, color: '#555' }}>{p.outcome}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })()}
        </div>
      )}

      {/* ── Clinical Atlas ─────────────────────────────────────────── */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          <div style={{ fontSize: 13, marginBottom: 16, color: '#555' }}>
            Gene-by-gene clinical reference — click any card to expand full details
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(380px,1fr))', gap: 16 }}>
            {Object.values(breakdown.gene_data).map(gd => {
              const col  = GENE_COLORS[gd.gene] || '#333';
              const open = expandedGene === gd.gene;
              return (
                <div key={gd.gene} style={{ background: '#fff', border: `2px solid ${col}44`,
                  borderRadius: 10, padding: 18, cursor: 'pointer' }}
                  onClick={() => setExpandedGene(open ? null : gd.gene)}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <span style={{ fontSize: 22, fontWeight: 900, color: col }}>{gd.gene}</span>
                      <Badge text={gd.inheritance} color={col} />
                      <Badge text={gd.locus} color={col} />
                    </div>
                    <span style={{ color: col, fontSize: 18 }}>{open ? '▲' : '▼'}</span>
                  </div>
                  <div style={{ fontSize: 11, color: '#666', marginTop: 6 }}>{gd.syndrome}</div>

                  {/* Key stats row */}
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 10 }}>
                    {[
                      ['n', gd.n],
                      ['Age', `${gd.mean_age_onset}yr`],
                      ['CD107a ✗', `${gd.cd107a_absent_n}/40`],
                      ['Ferritin', `${(gd.avg_ferritin_peak/1000).toFixed(1)}k`],
                      ['HSCT', `${gd.hsct_done_n}/40`],
                    ].map(([k, v]) => (
                      <div key={k} style={{ background: `${col}10`, border: `1px solid ${col}33`,
                        borderRadius: 6, padding: '4px 8px', fontSize: 10 }}>
                        <span style={{ color: '#888' }}>{k} </span>
                        <strong style={{ color: col }}>{v}</strong>
                      </div>
                    ))}
                  </div>

                  {open && (
                    <div style={{ marginTop: 14 }}>
                      <div style={{ marginBottom: 8, padding: '10px 12px', background: `${col}0a`,
                        borderRadius: 8, fontSize: 12 }}>
                        <strong style={{ color: col }}>Pathognomonic:</strong> {gd.pathognomonic}
                      </div>
                      <div style={{ marginBottom: 8, padding: '10px 12px', background: '#e8f5e9',
                        borderRadius: 8, fontSize: 12 }}>
                        <strong style={{ color: '#1b5e20' }}>Key Rule:</strong> {gd.key_rule}
                      </div>
                      <div style={{ marginBottom: 8, padding: '10px 12px', background: '#ffebee',
                        borderRadius: 8, fontSize: 12 }}>
                        <strong style={{ color: '#c62828' }}>Key Avoid:</strong> {gd.key_avoid}
                      </div>
                      <div style={{ padding: '10px 12px', background: '#e3f2fd',
                        borderRadius: 8, fontSize: 12 }}>
                        <strong style={{ color: '#1565c0' }}>Targeted Rx:</strong> {gd.targeted_rx}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ── Definitions ─────────────────────────────────────────────── */}
      {tab === 'Definitions' && definitions && (
        <div>
          {/* Key Clinical Distinctions */}
          <div style={{ background: '#fff', borderRadius: 10, padding: 20, marginBottom: 20,
            border: '2px solid #1a237e22', boxShadow: '0 1px 4px #0001' }}>
            <h3 style={{ marginTop: 0, color: '#1a237e', fontSize: 16 }}>
              🔑 Key Clinical Distinctions — Hereditary HLH Predisposition Atlas
            </h3>
            {definitions.key_clinical_distinctions?.map((d, i) => (
              <div key={i} style={{ marginBottom: 14, padding: '12px 14px', background: '#f5f5f5',
                borderRadius: 8, fontSize: 12, borderLeft: '4px solid #1a237e' }}>
                <strong style={{ color: '#1a237e' }}>{i + 1}. </strong>{d}
              </div>
            ))}
          </div>

          {/* Per-gene definition cards */}
          {definitions.definitions?.map(def => {
            const gene = def.term.split(' — ')[0];
            const col  = GENE_COLORS[gene] || '#333';
            const open = expandedDef === def.term;
            return (
              <div key={def.term} style={{ background: '#fff', border: `2px solid ${col}33`,
                borderRadius: 10, marginBottom: 12, overflow: 'hidden' }}>
                <div style={{ padding: '14px 18px', cursor: 'pointer', display: 'flex',
                  justifyContent: 'space-between', alignItems: 'center',
                  background: open ? `${col}0d` : '#fff' }}
                  onClick={() => setExpandedDef(open ? null : def.term)}>
                  <div>
                    <span style={{ fontSize: 16, fontWeight: 800, color: col }}>{gene}</span>
                    <span style={{ fontSize: 12, color: '#666', marginLeft: 10 }}>
                      {def.term.replace(gene + ' — ', '')}
                    </span>
                  </div>
                  <span style={{ color: col }}>{open ? '▲' : '▼'}</span>
                </div>
                {open && (
                  <div style={{ padding: '0 18px 16px', fontSize: 12, color: '#444',
                    whiteSpace: 'pre-wrap', lineHeight: 1.6, background: `${col}05` }}>
                    {def.definition}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
