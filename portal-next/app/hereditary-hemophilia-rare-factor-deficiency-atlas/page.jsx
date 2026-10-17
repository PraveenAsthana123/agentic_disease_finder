'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-hemophilia-rare-factor-deficiency-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'F8':       '#b71c1c',  // deep red            -- Hemophilia A; emicizumab; FEIBA-TMA; Intron-22
  'F9':       '#880e4f',  // deep crimson         -- Hemophilia B; Leyden-puberty-improvement; Hemgenix
  'VWF':      '#1565c0',  // deep blue            -- VWD; Type-2B-DDAVP-CI; Type-2N-mimics-HA
  'F11':      '#2e7d32',  // forest green         -- FXI; Ashkenazi; bleeding-NOT-proportional
  'F7':       '#e65100',  // burnt orange         -- FVII; PT-elevated-APTT-NORMAL; shortest-half-life
  'F13A1':    '#4a148c',  // deep purple          -- FXIII; umbilical-stump; ICH-highest; PT-APTT-TT-NORMAL
  'SERPINE1': '#006064',  // dark teal            -- PAI-1; delayed-wound-bleed; TXA-FIRST-LINE
  'ADAMTS13': '#33691e',  // olive green          -- Congenital TTP; MAHA; plasma-infusion-CURATIVE
};

const GENE_INFO = {
  'F8':       { full: 'Hemophilia-A / FVIII-deficiency / 1:5000-males / Intron-22-inversion-40-50%-severe-HA / FEIBA+emicizumab→TMA / Emicizumab-SC-prophylaxis-ALL-severe-HA / Roctavian-gene-therapy-FDA2023', locus: 'Xq28', size: '2351 aa / 280 kDa', inh: 'XLR LOF' },
  'F9':       { full: 'Hemophilia-B / FIX-deficiency / 1:25000-males / F9-Leyden-puberty-improvement-PATHOGNOMONIC / Anaphylaxis-FIX-concentrate+inhibitor / Hemgenix-FDA2022-gene-therapy', locus: 'Xq27.1', size: '461 aa / 57 kDa', inh: 'XLR LOF' },
  'VWF':      { full: 'VWD-Type-1/2A/2B/2M/2N/3 / Most-common-inherited-bleeding-1:100 / Type-2B-DDAVP-CI-ABSOLUTELY / Type-2N-mimics-mild-Hemophilia-A / ABO-O-VWF-25-35%-lower / Type-3-AR-complete-deficiency', locus: '12p13.31', size: '2813 aa / 360 kDa', inh: 'AD Type1/2; AR Type3' },
  'F11':      { full: 'FXI-deficiency / Hemophilia-C / Ashkenazi-1-in-190 / Bleeding-NOT-proportional-to-FXI-level / Fibrinolytic-rich-sites-high-risk / TXA-FIRST-LINE-pre-procedure', locus: '4q35.2', size: '625 aa / 80 kDa', inh: 'AR LOF' },
  'F7':       { full: 'FVII-deficiency / Most-common-rare-factor-deficiency-1:500k / PT-elevated+APTT-NORMAL-PATHOGNOMONIC / Shortest-half-life-4-6h / ICH-severe-<10% / rFVIIa-NovoSeven-licensed', locus: '13q34', size: '444 aa / 50 kDa', inh: 'AR LOF' },
  'F13A1':    { full: 'FXIII-A-deficiency / Umbilical-stump-bleeding-60-70%-PATHOGNOMONIC / ICH-HIGHEST-all-factor-deficiencies-25-30% / PT-APTT-TT-ALL-NORMAL-PATHOGNOMONIC / Urea-solubility-test / Prophylaxis-MANDATORY', locus: '6p24.3', size: '732 aa / 83 kDa', inh: 'AR LOF' },
  'SERPINE1': { full: 'PAI-1-deficiency / Delayed-wound-bleed-PATHOGNOMONIC / Postpartum-hemorrhage-PATHOGNOMONIC / Normal-coag-screen / Tranexamic-acid-FIRST-LINE / No-coag-factor-works', locus: '7q22.1', size: '402 aa / 52 kDa', inh: 'AR LOF biallelic' },
  'ADAMTS13': { full: 'Congenital-TTP / Upshaw-Schulman-syndrome / MAHA+thrombocytopenia-HALLMARK / Neonatal-jaundice+thrombocytopenia-PATHOGNOMONIC / Plasma-infusion-CURATIVE / NO-immunosuppression-needed', locus: '9q34.2', size: '1427 aa / 190 kDa', inh: 'AR LOF biallelic' },
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
        <Badge text={`Risk: ${(geneAtlas.bleeding_risk || '').split(';')[0]}`} color={bg} />
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
          ['Severe', cohort.severe_count != null ? `${cohort.severe_count}/40` : '—'],
          ['Hemarthrosis', cohort.hemarthrosis_count != null ? `${cohort.hemarthrosis_count}/40` : '—'],
          ['Inhibitor', cohort.inhibitor_count != null ? `${cohort.inhibitor_count}/40` : '—'],
          ['Prophylaxis', cohort.on_prophylaxis_count != null ? `${cohort.on_prophylaxis_count}/40` : '—'],
          gene === 'F13A1' ? ['ICH', cohort.intracranial_bleed_count != null ? `${cohort.intracranial_bleed_count}/40` : '—'] :
          gene === 'F13A1' ? ['Umbilical', cohort.umbilical_stump_count != null ? `${cohort.umbilical_stump_count}/40` : '—'] :
          gene === 'ADAMTS13' ? ['MAHA', cohort.maha_count != null ? `${cohort.maha_count}/40` : '—'] :
          gene === 'SERPINE1' ? ['DelayBleed', cohort.delayed_bleed_count != null ? `${cohort.delayed_bleed_count}/40` : '—'] :
          ['Family Hx', cohort.family_hx_count != null ? `${cohort.family_hx_count}/40` : '—'],
          ['Gene Rx', cohort.gene_therapy_count != null ? `${cohort.gene_therapy_count}/40` : '—'],
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

export default function HereditaryHemophiliaRareFactorDeficiencyAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => {
      setError(e.message);
      setLoading(false);
    });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading atlas…</div>;
  if (error) return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const genes = overview?.genes || [];

  return (
    <div style={{ maxWidth: 1200, margin: '0 auto', padding: '24px 16px', fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg, #b71c1c 0%, #880e4f 50%, #4a148c 100%)', color: '#fff', borderRadius: 12, padding: '24px 28px', marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>
          🩸 Hereditary Hemophilia & Rare Coagulation Factor Deficiency Atlas
        </h1>
        <div style={{ fontSize: 13, opacity: 0.9, marginTop: 6 }}>
          Complete 8-Gene Reference · F8-F9-VWF-F11-F7-F13A1-SERPINE1-ADAMTS13 · 320-Patient Aggregate (8×40, seeds 3686-3693)
        </div>
        <div style={{ display: 'flex', gap: 24, marginTop: 14, flexWrap: 'wrap' }}>
          {[
            ['320', 'Patients'],
            [overview?.total_severe, 'Severe'],
            [overview?.total_inhibitor, 'Inhibitors'],
            [overview?.total_hemarthrosis, 'Hemarthrosis'],
            [overview?.total_icb, 'ICH'],
            [overview?.total_maha, 'MAHA/TTP'],
          ].map(([val, label]) => (
            <div key={label} style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 24, fontWeight: 800 }}>{val ?? '—'}</div>
              <div style={{ fontSize: 11, opacity: 0.8 }}>{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#b71c1c' : '#eee', color: tab === t ? '#fff' : '#333',
            fontWeight: tab === t ? 700 : 400, fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {/* Overview Tab */}
      {tab === 'Overview' && (
        <div>
          {/* Gene grid */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 12, marginBottom: 24 }}>
            {genes.map(gene => {
              const gs = overview?.gene_summaries?.find(g => g.gene === gene) || {};
              const color = GENE_COLORS[gene] || '#555';
              const geneAtlas = overview?.atlas_genes?.find(g => g.gene === gene) || {};
              return (
                <div key={gene} style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 14 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                    <div style={{ background: color, color: '#fff', borderRadius: 5, padding: '4px 10px', fontWeight: 700, fontSize: 15 }}>{gene}</div>
                    <div style={{ fontSize: 11, color: '#666' }}>{GENE_INFO[gene]?.locus} · {GENE_INFO[gene]?.inh}</div>
                  </div>
                  <div style={{ fontSize: 11, color: '#333', marginBottom: 6 }}>{GENE_INFO[gene]?.full}</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 4 }}>
                    {[
                      ['N', gs.n], ['Severe', gs.severe_count], ['ICH', gs.intracranial_bleed_count ?? 0],
                    ].map(([l, v]) => (
                      <div key={l} style={{ textAlign: 'center', background: '#f5f5f5', borderRadius: 3, padding: '3px 0' }}>
                        <div style={{ fontWeight: 700, fontSize: 13 }}>{v}</div>
                        <div style={{ fontSize: 10, color: '#888' }}>{l}</div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Key Clinical Rules */}
          <div style={{ background: '#fff8e1', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 10px' }}>⚡ Key Clinical Rules — Memorize</h3>
            {(overview?.key_clinical_rules || []).map((rule, i) => (
              <div key={i} style={{ padding: '6px 0', borderBottom: '1px solid #ffe082', fontSize: 13 }}>
                <b>{rule.split(':')[0]}:</b> {rule.split(':').slice(1).join(':')}
              </div>
            ))}
          </div>

          {/* Hemostasis disorder categories */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div style={{ background: '#ffebee', borderRadius: 8, padding: 16 }}>
              <h4 style={{ margin: '0 0 8px', color: '#b71c1c' }}>🔴 X-Linked Hemophilias</h4>
              <div style={{ fontSize: 12 }}>
                <b>F8 (Hemophilia A):</b> 1:5000 males; most common severe inherited bleeding; Emicizumab prophylaxis<br/>
                <b>F9 (Hemophilia B):</b> 1:25000 males; Leyden variant improves post-puberty; Hemgenix FDA2022<br/>
                <b>Critical:</b> FEIBA + emicizumab → TMA; always use rFVIIa instead
              </div>
            </div>
            <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16 }}>
              <h4 style={{ margin: '0 0 8px', color: '#1565c0' }}>🔵 von Willebrand Disease</h4>
              <div style={{ fontSize: 12 }}>
                <b>VWF (VWD):</b> Most common inherited bleeding 1:100; 6 subtypes<br/>
                <b>Type 2B:</b> DDAVP CONTRAINDICATED ABSOLUTELY — worsens thrombocytopenia<br/>
                <b>Type 2N:</b> Mimics mild Hemophilia A — test VWF FVIII-binding assay
              </div>
            </div>
            <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 16 }}>
              <h4 style={{ margin: '0 0 8px', color: '#2e7d32' }}>🟢 Rare Autosomal Factor Deficiencies</h4>
              <div style={{ fontSize: 12 }}>
                <b>F11:</b> Ashkenazi 1:190; bleeding NOT proportional to level; TXA pre-procedure<br/>
                <b>F7:</b> PT elevated + APTT NORMAL; rFVIIa licensed; shortest half-life<br/>
                <b>F13A1:</b> PT/APTT/TT ALL NORMAL; ICH highest risk; prophylaxis mandatory
              </div>
            </div>
            <div style={{ background: '#ede7f6', borderRadius: 8, padding: 16 }}>
              <h4 style={{ margin: '0 0 8px', color: '#4a148c' }}>🟣 Fibrinolysis / TTP Disorders</h4>
              <div style={{ fontSize: 12 }}>
                <b>SERPINE1 (PAI-1):</b> Delayed wound bleeding; TXA FIRST-LINE; no coag factor works<br/>
                <b>ADAMTS13 (cTTP):</b> MAHA + thrombocytopenia; plasma infusion CURATIVE; no immunosuppression<br/>
                <b>Neonatal TTP:</b> Jaundice + thrombocytopenia in newborn = congenital TTP PATHOGNOMONIC
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {tab === 'Gene Table' && (
        <div>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#b71c1c', color: '#fff' }}>
                {['Gene', 'Locus', 'Protein', 'Inheritance', 'Bleeding Risk', 'Pathognomonic', 'Targeted Rx'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(overview?.atlas_genes || []).map((g, i) => (
                <tr key={g.gene} style={{ background: i % 2 === 0 ? '#fff' : '#fafafa', verticalAlign: 'top' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[g.gene] || '#333' }}>{g.gene}</td>
                  <td style={{ padding: '8px 10px' }}>{g.locus}</td>
                  <td style={{ padding: '8px 10px', maxWidth: 180 }}>{GENE_INFO[g.gene]?.size}<br/><span style={{ color: '#888', fontSize: 10 }}>{GENE_INFO[g.gene]?.inh}</span></td>
                  <td style={{ padding: '8px 10px' }}>{g.inheritance}</td>
                  <td style={{ padding: '8px 10px', maxWidth: 180 }}>{g.bleeding_risk}</td>
                  <td style={{ padding: '8px 10px', maxWidth: 200, color: '#e65100', fontWeight: 500 }}>{g.pathognomonic}</td>
                  <td style={{ padding: '8px 10px', maxWidth: 180 }}>{g.targeted_rx}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Clinical Atlas Tab */}
      {tab === 'Clinical Atlas' && (
        <div>
          {genes.map(gene => (
            <GeneCard
              key={gene}
              gene={gene}
              color={GENE_COLORS[gene] || '#555'}
              info={GENE_INFO[gene]}
              breakdown={breakdown}
            />
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {tab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ marginBottom: 12 }}>Clinical Definitions & Cascade Testing</h3>
          {/* Abbreviations */}
          <div style={{ background: '#f5f5f5', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h4 style={{ margin: '0 0 10px' }}>Key Abbreviations</h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 4 }}>
              {Object.entries(definitions.key_abbreviations || {}).map(([abbr, def]) => (
                <div key={abbr} style={{ fontSize: 12, padding: '3px 0' }}>
                  <b style={{ color: '#b71c1c' }}>{abbr}:</b> {def}
                </div>
              ))}
            </div>
          </div>
          {/* Cascade Testing */}
          <div style={{ background: '#e8f5e9', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h4 style={{ margin: '0 0 10px' }}>Cascade Testing Protocols</h4>
            {Object.entries(definitions.cascade_testing || {}).map(([key, val]) => (
              <div key={key} style={{ marginBottom: 8, fontSize: 12 }}>
                <b style={{ color: '#2e7d32', textTransform: 'capitalize' }}>{key.replace(/_/g, ' ')}:</b> {val}
              </div>
            ))}
          </div>
          {/* Per-gene definitions */}
          {genes.map(gene => {
            const gd = definitions.definitions?.[gene];
            if (!gd) return null;
            return (
              <div key={gene} style={{ border: `2px solid ${GENE_COLORS[gene] || '#555'}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[gene], fontSize: 16, marginBottom: 6 }}>{gene}</div>
                <div style={{ fontSize: 11, color: '#555', fontFamily: 'monospace', whiteSpace: 'pre-wrap', maxHeight: 200, overflowY: 'auto', background: '#f9f9f9', padding: 8, borderRadius: 4 }}>
                  {gd.protein_size}
                </div>
                <div style={{ marginTop: 8, fontSize: 12 }}>
                  <b>Surveillance:</b> {gd.surveillance}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
