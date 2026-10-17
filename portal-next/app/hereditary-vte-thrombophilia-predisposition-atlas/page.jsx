'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-vte-thrombophilia-predisposition-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'F5':       '#1a237e',  // deep indigo       -- FVL-APC-resistance; OCP-MULTIPLICATIVE; APCR-ratio
  'F2':       '#880e4f',  // deep crimson       -- Prothrombin-G20210A; NOT-standard-coag; CVST
  'PROC':     '#004d40',  // deep teal          -- Protein-C; NPF-AR-PATHOGNOMONIC; warfarin-skin-necrosis
  'PROS1':    '#bf360c',  // burnt orange       -- Protein-S; PROS2-pseudogene-WES-MANDATORY
  'SERPINC1': '#4a148c',  // deep purple        -- Antithrombin-III; most-severe; heparin-resistance-II-HBS
  'THBD':     '#006064',  // dark cyan          -- Thrombomodulin; arterial-young-MI; coronary-NORMAL-PATHOGNOMONIC
  'PLG':      '#33691e',  // olive green        -- Plasminogen; ligneous-conjunctivitis-PATHOGNOMONIC; Ryplazim
  'FGG':      '#e65100',  // deep amber         -- Fibrinogen-Gamma; thrombophilic-dysfibrinogenemia; TCT-prolonged
};

const GENE_INFO = {
  'F5':       { full: 'Factor-V-Leiden-R506Q / APC-Resistance / 5%-Europeans-most-common / OCP+FVL-MULTIPLICATIVE-30-35x / APCR-ratio-screen / DOAC-preferred', locus: '7q35', size: '2224 aa / 330 kDa', inh: 'AD semi-dominant LOF' },
  'F2':       { full: 'Prothrombin-G20210A / Elevated-prothrombin-130% / 2nd-most-common-1-3% / NOT-detectable-PT-APTT / CVST-enriched / DOAC-preferred', locus: '11p11.2', size: '622 aa / 72 kDa', inh: 'AD GOF 3-prime-UTR' },
  'PROC':     { full: 'Protein-C-Deficiency-Type-I-II / NPF-AR-within-72h-PATHOGNOMONIC / Warfarin-skin-necrosis-PATHOGNOMONIC / LMWH-bridge-ALWAYS / Ceprotin-NPF / DOAC-preferred', locus: '2q14.3', size: '461 aa / 62 kDa', inh: 'AR homozygous NPF; AD heterozygous VTE' },
  'PROS1':    { full: 'Protein-S-Alpha-Deficiency-Type-I-II-III / PROS2-pseudogene-WES-MANDATORY / NEVER-diagnose-pregnancy-OCP / Warfarin-skin-necrosis-risk / DOAC-preferred', locus: '3q11.2', size: '676 aa / 75 kDa', inh: 'AD LOF heterozygous' },
  'SERPINC1': { full: 'Antithrombin-III-Deficiency-Type-I-Type-II-HBS / Most-severe-hereditary-thrombophilia / Type-II-HBS-Heparin-Resistance-PATHOGNOMONIC / AT-concentrate-MANDATORY / NEVER-escalate-UFH-blindly', locus: '1q25.1', size: '432 aa / 58 kDa', inh: 'AD LOF; homozygous lethal' },
  'THBD':     { full: 'Thrombomodulin-Deficiency / Arterial-thrombosis-young-MI-stroke / Normal-coronary-angiogram-PATHOGNOMONIC / Standard-coag-NORMAL / Protein-C-level-NORMAL / Statins-aspirin', locus: '20p11.21', size: '575 aa / 75 kDa', inh: 'AD LOF heterozygous' },
  'PLG':      { full: 'Plasminogen-Deficiency-Type-I / Ligneous-conjunctivitis-PATHOGNOMONIC / Tranexamic-acid-CONTRAINDICATED-ABSOLUTELY / Ryplazim-FDA2021 / Fibrinolysis-ABSENT / DOAC-LMWH', locus: '6q26', size: '810 aa / 92 kDa', inh: 'AR biallelic Type-I; AD partial Type-II' },
  'FGG':      { full: 'Thrombophilic-Dysfibrinogenemia-FGG-R301C-Bicetre-I / TCT-prolonged-PATHOGNOMONIC / Clauss-ratio-less-0.7-PATHOGNOMONIC / ARTERIAL+VENOUS-both / Fibrinogen-concentrate-CI-thrombophilic', locus: '4q32.1', size: '453 aa / 47 kDa', inh: 'AD heterozygous GOF/LOF paradox' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, breakdown }) {
  const bg = color;
  const cohort = breakdown?.gene_data?.[gene] || {};
  const geneAtlas = breakdown?.atlas_genes?.find(g => g.gene === gene) || {};
  return (
    <div style={{ border: `2px solid ${bg}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: bg, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div>
          <div style={{ fontWeight: 600 }}>{geneAtlas.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Badge text={`Thrombosis: ${(geneAtlas.thrombosis_risk || '').split(';')[0]}`} color={bg} />
      </div>
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      {geneAtlas.key_rule && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ✓ <b>Key rule:</b> {geneAtlas.key_rule}
        </div>
      )}
      {geneAtlas.key_avoid && (
        <div style={{ fontSize: 12, background: '#ffebee', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ✗ <b>Avoid:</b> {geneAtlas.key_avoid}
        </div>
      )}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 8, marginTop: 10 }}>
        {[
          ['N', cohort.n],
          ['Avg age', cohort.mean_age ? cohort.mean_age + 'yr' : '—'],
          ['DVT/PE', cohort.dvt_pe_count != null ? `${cohort.dvt_pe_count}/40` : '—'],
          ['Arterial', cohort.arterial_event_count != null ? `${cohort.arterial_event_count}/40` : '—'],
          ['Recurrent VTE', cohort.recurrent_vte_count != null ? `${cohort.recurrent_vte_count}/40` : '—'],
          ['On anticoag', cohort.on_anticoag_count != null ? `${cohort.on_anticoag_count}/40` : '—'],
          ['Family hx', cohort.family_hx_count != null ? `${cohort.family_hx_count}/40` : '—'],
          gene === 'PLG' ? ['Ligneous', cohort.ligneous_count != null ? `${cohort.ligneous_count}/40` : '—'] :
          gene === 'SERPINC1' ? ['Hep resist', cohort.heparin_resistance_count != null ? `${cohort.heparin_resistance_count}/40` : '—'] :
          (gene === 'PROC' || gene === 'PROS1') ? ['Warf necr', cohort.warfarin_necrosis_count != null ? `${cohort.warfarin_necrosis_count}/40` : '—'] :
          ['Targeted Rx', '—'],
        ].map(([label, val]) => (
          <div key={label} style={{ textAlign: 'center', background: '#f5f5f5', borderRadius: 4, padding: '4px 2px' }}>
            <div style={{ fontSize: 15, fontWeight: 700 }}>{val}</div>
            <div style={{ fontSize: 10, color: '#888' }}>{label}</div>
          </div>
        ))}
      </div>
      {geneAtlas.targeted_rx && (
        <div style={{ fontSize: 11, color: '#555', marginTop: 8, borderTop: '1px solid #eee', paddingTop: 6 }}>
          💊 <b>Rx:</b> {geneAtlas.targeted_rx}
        </div>
      )}
    </div>
  );
}

export default function HereditaryVTEThrombophiliaAtlasPage() {
  const [tab, setTab] = useState('Overview');
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
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  const summ = overview?.clinical_summary || {};

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 1100, margin: '0 auto', padding: 24 }}>
      <div style={{ background: '#1a237e', color: '#fff', borderRadius: 10, padding: '20px 28px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>🧬 Hereditary VTE / Thrombophilia Predisposition Atlas</h1>
        <div style={{ fontSize: 13, opacity: 0.85, marginTop: 4 }}>
          Complete 8-Gene Reference · F5 · F2 · PROC · PROS1 · SERPINC1 · THBD · PLG · FGG
          · 320-Patient Aggregate · Seeds 3678-3685
        </div>
        <div style={{ fontSize: 12, opacity: 0.75, marginTop: 2 }}>
          Factor-V-Leiden · Prothrombin-G20210A · Protein-C · Protein-S · Antithrombin-III
          · Thrombomodulin · Plasminogen · Fibrinogen-Gamma
        </div>
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '7px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#1a237e' : '#e3e8f0', color: tab === t ? '#fff' : '#333',
            fontWeight: tab === t ? 700 : 400
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ textAlign: 'center', padding: 40, color: '#888' }}>Loading atlas data…</div>}
      {error && <div style={{ background: '#ffebee', padding: 12, borderRadius: 6, color: '#c62828' }}>Error: {error}</div>}

      {!loading && !error && tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14, marginBottom: 22 }}>
            {[
              ['Total Patients', overview.total_patients, '#1a237e'],
              ['Genes', overview.n_genes, '#880e4f'],
              ['Mean Age', overview.mean_age_overall + ' yr', '#004d40'],
              ['DVT / PE Events', summ.dvt_pe_total, '#4a148c'],
              ['Arterial Events', summ.arterial_event_total, '#006064'],
              ['Recurrent VTE', summ.recurrent_vte_total, '#bf360c'],
              ['On Anticoagulation', summ.on_anticoag_total, '#33691e'],
              ['Heparin Resistance (SERPINC1)', summ.heparin_resistance_total, '#e65100'],
            ].map(([label, val, bg]) => (
              <div key={label} style={{ background: bg, color: '#fff', borderRadius: 8, padding: '14px 16px', textAlign: 'center' }}>
                <div style={{ fontSize: 26, fontWeight: 800 }}>{val}</div>
                <div style={{ fontSize: 11, opacity: 0.85, marginTop: 2 }}>{label}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <div style={{ background: '#ffebee', borderRadius: 8, padding: 16 }}>
              <h3 style={{ margin: '0 0 10px', fontSize: 14, color: '#c62828' }}>⚠ Special Alerts</h3>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
                <li>Warfarin skin necrosis (PROC+PROS1): <b>{summ.warfarin_necrosis_total}</b> events — ALWAYS bridge LMWH</li>
                <li>Heparin resistance Type-II-HBS (SERPINC1): <b>{summ.heparin_resistance_total}</b> — switch AT-concentrate/argatroban</li>
                <li>Ligneous conjunctivitis (PLG Type-I): <b>{summ.ligneous_conjunctivitis_total}</b> — Ryplazim FDA-2021</li>
                <li>THBD: arterial thrombosis predominant — normal coronary angiogram PATHOGNOMONIC</li>
              </ul>
            </div>
            <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 16 }}>
              <h3 style={{ margin: '0 0 10px', fontSize: 14, color: '#2e7d32' }}>✓ Universal Rules</h3>
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13 }}>
                <li>DOAC preferred over warfarin for ALL 8 genes</li>
                <li>F5+F2: test genetically (not coag panels) — can test anytime</li>
                <li>PROC/PROS1/AT: test OFF heparin/warfarin/OCP and NOT in acute VTE</li>
                <li>PROS1: NEVER diagnose in pregnancy (physiological 50% PS fall)</li>
                <li>Cascade test all first-degree relatives ≥16-18yr</li>
              </ul>
            </div>
          </div>

          <div style={{ background: '#f5f5f5', borderRadius: 8, padding: 16, marginBottom: 16 }}>
            <h3 style={{ margin: '0 0 10px', fontSize: 14 }}>Thrombophilia Severity Hierarchy</h3>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {['SERPINC1 Type-I (most severe)', 'PROC homozygous (NPF)', 'PROS1 Type-I', 'SERPINC1 Type-II-HBS',
                'F5 homozygous', 'PROC heterozygous', 'PROS1', 'F5 heterozygous ≈ F2'].map((item, i) => (
                <div key={item} style={{ background: `hsl(${220 + i * 15},60%,${35 + i * 5}%)`, color: '#fff', borderRadius: 16, padding: '4px 12px', fontSize: 12, fontWeight: 600 }}>
                  {i + 1}. {item}
                </div>
              ))}
            </div>
          </div>

          <div style={{ background: '#fffde7', borderRadius: 8, padding: 16 }}>
            <h3 style={{ margin: '0 0 10px', fontSize: 14 }}>Per-Gene Summary</h3>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ background: '#1a237e', color: '#fff' }}>
                    {['Gene', 'Locus', 'N', 'Avg Age', 'DVT/PE%', 'Arterial%', 'Recurrent%', 'On Anticoag%', 'Key Rule'].map(h => (
                      <th key={h} style={{ padding: '6px 8px', textAlign: 'left' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {(overview.per_gene || []).map((g, i) => (
                    <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f5f5f5' }}>
                      <td style={{ padding: '5px 8px', fontWeight: 700, color: GENE_COLORS[g.gene] }}>{g.gene}</td>
                      <td style={{ padding: '5px 8px', fontSize: 11 }}>{g.locus}</td>
                      <td style={{ padding: '5px 8px' }}>{g.n}</td>
                      <td style={{ padding: '5px 8px' }}>{g.mean_age}yr</td>
                      <td style={{ padding: '5px 8px' }}>{g.dvt_pe_pct}%</td>
                      <td style={{ padding: '5px 8px', color: g.arterial_event_pct >= 30 ? '#c62828' : 'inherit', fontWeight: g.arterial_event_pct >= 30 ? 700 : 400 }}>{g.arterial_event_pct}%</td>
                      <td style={{ padding: '5px 8px' }}>{g.recurrent_vte_pct}%</td>
                      <td style={{ padding: '5px 8px' }}>{g.on_anticoag_pct}%</td>
                      <td style={{ padding: '5px 8px', fontSize: 10, color: '#555' }}>{(g.key_rule || '').substring(0, 60)}…</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {!loading && !error && tab === 'Gene Table' && breakdown && (
        <div>
          <h2 style={{ fontSize: 17, marginBottom: 16 }}>8-Gene Reference Table</h2>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1a237e', color: '#fff' }}>
                  {['Gene', 'Locus', 'Syndrome', 'Inheritance', 'Thrombosis Risk', 'Pathognomonic', 'Key Rule', 'Avoid', 'Targeted Rx'].map(h => (
                    <th key={h} style={{ padding: '7px 8px', textAlign: 'left', fontSize: 11 }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(breakdown.atlas_genes || []).map((g, i) => (
                  <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#f8f8f8', verticalAlign: 'top' }}>
                    <td style={{ padding: '6px 8px', fontWeight: 800, color: GENE_COLORS[g.gene], fontSize: 13 }}>{g.gene}</td>
                    <td style={{ padding: '6px 8px', fontSize: 11 }}>{g.locus}</td>
                    <td style={{ padding: '6px 8px', fontSize: 11, maxWidth: 160 }}>{g.syndrome}</td>
                    <td style={{ padding: '6px 8px', fontSize: 10 }}>{g.inheritance}</td>
                    <td style={{ padding: '6px 8px', fontSize: 10, maxWidth: 130 }}>{g.thrombosis_risk}</td>
                    <td style={{ padding: '6px 8px', fontSize: 10, background: '#fff8e1', maxWidth: 140 }}>{g.pathognomonic}</td>
                    <td style={{ padding: '6px 8px', fontSize: 10, background: '#e8f5e9', maxWidth: 140 }}>{g.key_rule}</td>
                    <td style={{ padding: '6px 8px', fontSize: 10, background: '#ffebee', maxWidth: 130 }}>{g.key_avoid}</td>
                    <td style={{ padding: '6px 8px', fontSize: 10, maxWidth: 130 }}>{g.targeted_rx}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {!loading && !error && tab === 'Clinical Atlas' && breakdown && (
        <div>
          <h2 style={{ fontSize: 17, marginBottom: 16 }}>Clinical Gene Atlas — VTE / Thrombophilia</h2>
          {Object.keys(GENE_COLORS).map(gene => (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} breakdown={breakdown} />
          ))}
        </div>
      )}

      {!loading && !error && tab === 'Definitions' && definitions && (
        <div>
          <h2 style={{ fontSize: 17, marginBottom: 16 }}>Clinical Definitions</h2>
          {(definitions.definitions || []).map((d, i) => (
            <div key={i} style={{ marginBottom: 12, borderBottom: '1px solid #eee', paddingBottom: 10 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#1a237e', marginBottom: 3 }}>{d.term}</div>
              <div style={{ fontSize: 12, color: '#444', lineHeight: 1.55 }}>{d.definition}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
