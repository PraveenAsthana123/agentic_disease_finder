'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-nkh-glycine-encephalopathy-hyperekplexia-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'GLDC':   '#1a237e',  // dark indigo        -- P-protein; 80% NKH; burst-suppression; hiccupping PATHOGNOMONIC
  'AMT':    '#283593',  // indigo-700         -- T-protein; 10% NKH; biochemically identical to GLDC
  'GCSH':   '#303f9f',  // indigo-600         -- H-protein; <1% NKH; ultra-rare; carrier protein
  'DLD':    '#4a148c',  // deep purple        -- L-protein; COMBINED PDH+KGDH+BCKDH+GCS; lactic acidosis PATHOGNOMONIC
  'GLRA1':  '#b71c1c',  // dark red           -- Glycine receptor α1; Hyperekplexia 1; startle disease; nose-tap PATHOGNOMONIC
  'GLRB':   '#c62828',  // red-800            -- Glycine receptor β; Hyperekplexia 2; identical to GLRA1 clinically
  'SLC6A5': '#e65100',  // deep orange        -- GlyT2; Hyperekplexia 3; presynaptic glycine depletion
  'SLC6A9': '#2e7d32',  // dark green         -- GlyT1; Glycine encephalopathy; CSF glycine elevated; GCS NORMAL
};

const GENE_INFO = {
  'GLDC':   { full: 'GLDC-1020aa / AR / Glycine-decarboxylase-P-protein / GCS-step-1 / 9p24.1 / Classical-NKH-80pct / CSF:plasma-ratio-GT0.08-PATHOGNOMONIC / Burst-suppression-EEG / Hiccupping-DISTINCTIVE / Sodium-benzoate+dextromethorphan / PLP-dependent / Vigabatrin-AVOID', locus: '9p24.1', size: '1020 aa / 115 kDa', inh: 'Autosomal recessive (AR; PLP-dependent; 80% of NKH; pyridoxine trial ~20% response; sodium benzoate + dextromethorphan)' },
  'AMT':    { full: 'AMT-403aa / AR / Aminomethyltransferase-T-protein / GCS-step-2 / 3p21.31 / NKH-10pct / IDENTICAL-biochemistry-GLDC / More-attenuated-variants / Molecular-panel-MANDATORY / p.G269D-Japanese-founder', locus: '3p21.31', size: '403 aa / 45 kDa', inh: 'Autosomal recessive (AR; T-protein GCS step 2; biochemically identical to GLDC; gene panel mandatory)' },
  'GCSH':   { full: 'GCSH-173aa / AR / H-protein-lipoamide-carrier / GCS-carrier / 16q23.2 / NKH-LT1pct / ULTRA-RARE-LT50-cases / Biochemically-GLDC-IDENTICAL / Smallest-GCS-protein / MLPA-required', locus: '16q23.2', size: '173 aa / 20 kDa', inh: 'Autosomal recessive (AR; lipoamide-bearing H-protein carrier; <1% NKH; <50 cases worldwide; MLPA for deletions)' },
  'DLD':    { full: 'DLD-509aa / AR / Dihydrolipoamide-dehydrogenase-L-protein-E3 / COMBINED-PDH+KGDH+BCKDH+GCS / 7q31.1 / NKH-type-III-PLUS-lactic-acidosis-PATHOGNOMONIC / Ashkenazi-p.G229C-founder / Liver-dominant / DCA-thiamine-riboflavin / Liver-Tx-NOT-curative', locus: '7q31.1', size: '509 aa / 56 kDa', inh: 'Autosomal recessive (AR; E3 subunit shared by 4 complexes; NKH + lactic acidosis PATHOGNOMONIC; Ashkenazi founder p.G229C)' },
  'GLRA1':  { full: 'GLRA1-464aa / AD-dominant-negative-OR-AR / Glycine-receptor-alpha1 / Cl-channel / 5q33.1 / Hyperekplexia-type-1-startle-disease / Nose-tap-PATHOGNOMONIC / Glycine-NORMAL / Hypertonia-NOT-hypotonia / Clonazepam-HIGHLY-EFFECTIVE / p.R271L-most-common-worldwide', locus: '5q33.1', size: '464 aa / 52 kDa', inh: 'Autosomal dominant (dominant negative ~60%) or AR; glycine receptor α1; hyperekplexia — glycine levels NORMAL' },
  'GLRB':   { full: 'GLRB-474aa / AR / Glycine-receptor-beta / Gephyrin-anchor / 4q32.1 / Hyperekplexia-type-2 / IDENTICAL-phenotype-GLRA1 / Glycine-NORMAL / Clonazepam-HIGHLY-EFFECTIVE / GLRA1-GLRB-panel-MANDATORY', locus: '4q32.1', size: '474 aa / 53 kDa', inh: 'Autosomal recessive (AR only; β-subunit anchors via gephyrin; hyperekplexia identical to GLRA1; glycine NORMAL)' },
  'SLC6A5': { full: 'SLC6A5-799aa / AD-haploinsufficiency-OR-AR / GlyT2-glycine-transporter-2 / Presynaptic-reuptake / 11p15.1 / Hyperekplexia-type-3 / Presynaptic-glycine-depleted / Glycine-NORMAL-or-mildly-elevated / Clonazepam-EFFECTIVE / p.Y705S-p.N509S-common-AD', locus: '11p15.1', size: '799 aa / 90 kDa', inh: 'AD (haploinsufficiency) or AR; GlyT2 presynaptic glycine recycling; hyperekplexia — glycine NORMAL' },
  'SLC6A9': { full: 'SLC6A9-706aa / AR / GlyT1-glycine-transporter-1 / Astrocytic-synaptic-clearance / 1p34.1 / GlyT1-encephalopathy / CSF-glycine-ELEVATED-mimics-NKH / GCS-enzyme-NORMAL-KEY-DDx / NMDA-modulators / LT20-cases-worldwide', locus: '1p34.1', size: '706 aa / 80 kDa', inh: 'Autosomal recessive (AR; astrocytic GlyT1; CSF glycine elevated but GCS enzyme NORMAL — DDx from true NKH)' },
};

const PHENOTYPE_ROWS = [
  { gene: 'GLDC',   group: 'Classical NKH (GCS P-protein, 80%)',              glycine_ratio: '>0.08',   nbs: '✗',  clonaz: '✗',  la: '—',  rx: 'Sodium benzoate + dextromethorphan ± pyridoxine', unique: 'CSF:plasma glycine ratio >0.08 PATHOGNOMONIC; burst-suppression EEG at birth; hiccupping DISTINCTIVE; profound hypotonia (NOT hypertonia); vigabatrin AVOID; GCS PLP-dependent enzyme; 80% of all NKH' },
  { gene: 'AMT',    group: 'Classical NKH (GCS T-protein, 10%)',              glycine_ratio: '>0.08',   nbs: '✗',  clonaz: '✗',  la: '—',  rx: 'Sodium benzoate + dextromethorphan ± pyridoxine', unique: 'BIOCHEMICALLY IDENTICAL to GLDC — gene panel (GLDC/AMT/GCSH) MANDATORY; more attenuated variants than GLDC; p.G269D Japanese founder → intermediate phenotype; treatment identical to GLDC' },
  { gene: 'GCSH',   group: 'Classical NKH (GCS H-protein, <1%)',              glycine_ratio: '>0.08',   nbs: '✗',  clonaz: '✗',  la: '—',  rx: 'Sodium benzoate + dextromethorphan', unique: 'ULTRA-RARE: <50 confirmed cases worldwide; H-protein is lipoamide-bearing carrier (smallest GCS protein, 173aa/20kDa); biochemically indistinguishable from GLDC/AMT; MLPA required for deletion detection; some attenuated forms' },
  { gene: 'DLD',    group: 'NKH type III + COMBINED deficiency',              glycine_ratio: '>0.08',   nbs: '±',  clonaz: '✗',  la: '↑↑', rx: 'Thiamine + riboflavin + DCA + benzoate + BCAA-reduced', unique: 'PATHOGNOMONIC: NKH glycine elevation PLUS lactic acidosis simultaneously (PDH + KGDH + BCKDH + GCS all affected); Ashkenazi p.G229C founder → liver-dominant; BCAA/alloisoleucine elevated (MSUD-III component); liver Tx NOT curative (brain/heart DLD affected)' },
  { gene: 'GLRA1',  group: 'Hyperekplexia type 1 (startle disease)',          glycine_ratio: 'NORMAL',  nbs: '✗',  clonaz: '✓',  la: '—',  rx: 'Clonazepam (highly effective; lifelong)', unique: 'PATHOGNOMONIC: exaggerated non-habituating startle → tonic spasm; NOSE TAP TEST: tap nasal bridge → sustained stiffening = PATHOGNOMONIC; HYPERTONIA (NOT hypotonia like NKH); glycine levels NORMAL; p.R271L most common worldwide; umbilicus manoeuvre terminates neonatal spell; AD dominant negative or AR' },
  { gene: 'GLRB',   group: 'Hyperekplexia type 2',                           glycine_ratio: 'NORMAL',  nbs: '✗',  clonaz: '✓',  la: '—',  rx: 'Clonazepam (highly effective; lifelong)', unique: 'IDENTICAL phenotype to GLRA1 hyperekplexia; AR only (no dominant GLRB variants); β-subunit required for gephyrin anchoring at synapses; molecular panel (GLRA1+GLRB+SLC6A5) required to distinguish; rarer than GLRA1' },
  { gene: 'SLC6A5', group: 'Hyperekplexia type 3 (GlyT2)',                   glycine_ratio: '±normal', nbs: '✗',  clonaz: '✓',  la: '—',  rx: 'Clonazepam (effective; lifelong)', unique: 'GlyT2 PRESYNAPTIC glycine recycler — LOF depletes presynaptic glycine stores → glycinergic inhibition lost; AD (haploinsufficiency) or AR; same clinical phenotype as GLRA1/GLRB; glycine NORMAL or mildly elevated (transporter less efficient); p.Y705S common AD variant' },
  { gene: 'SLC6A9', group: 'GlyT1 encephalopathy (mimics NKH)',              glycine_ratio: 'elevated', nbs: '✗',  clonaz: '✗',  la: '—',  rx: 'Symptomatic; NMDA modulators; sarcosine; NOT NaB alone', unique: 'GlyT1 ASTROCYTIC synaptic clearance — LOF → synaptic glycine accumulates → CSF glycine elevated (mimics NKH!); KEY DDx: GCS enzyme activity NORMAL in fibroblasts/liver; <20 cases worldwide; NOT hyperekplexia; burst-suppression + apnea (NKH-like neonatal encephalopathy); sarcosine (GlyT1 substrate competitor) may help' },
];

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, cohort }) {
  const groupLabel = {
    GLDC:   'NKH-P-protein-80pct-GCS-PLP-BurstSuppression-Hiccupping-NaBenzoate-Dextromethorphan',
    AMT:    'NKH-T-protein-10pct-IDENTICAL-GLDC-Attenuated-Variants-p.G269D-Japanese',
    GCSH:   'NKH-H-protein-LT1pct-ULTRARARE-LipoamideCarrier-MLPA',
    DLD:    'NKH-III-CombinedPDH-KGDH-BCKDH-GCS-LacticAcidosis-PATHOGNOMONIC-AshkenaziFounder',
    GLRA1:  'Hyperekplexia1-StarTleDisease-NoseTap-PATHOGNOMONIC-ClonazepamEffective-p.R271L',
    GLRB:   'Hyperekplexia2-IDENTICAL-GLRA1-AR-Gephyrin-Anchor',
    SLC6A5: 'Hyperekplexia3-GlyT2-PresynapticGlycineDepletion-ClonazepamEffective',
    SLC6A9: 'GlyT1-Encephalopathy-CSFglycineElevated-GCS-NORMAL-DDx-NKH-Mimics',
  }[gene] || gene;

  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 12, marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ background: color, color: '#fff', fontWeight: 700, padding: '3px 10px', borderRadius: 5, fontSize: 15 }}>{gene}</span>
          <span style={{ marginLeft: 10, fontSize: 12, color: '#555' }}>{groupLabel}</span>
        </div>
        <span style={{ fontSize: 11, color: '#888' }}>seed {cohort?.seed ?? '—'} · n={cohort?.n ?? 40}</span>
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: '#333' }}>{info?.full}</div>
      <div style={{ marginTop: 6, fontSize: 12 }}>
        <strong>Locus:</strong> {info?.locus} &nbsp;|&nbsp;
        <strong>Size:</strong> {info?.size} &nbsp;|&nbsp;
        <strong>Inheritance:</strong> {info?.inh}
      </div>
      {cohort && (
        <div style={{ marginTop: 6, fontSize: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`CSF:plasma ratio: ${cohort.glycine_csf_plasma_ratio ?? '—'}`} color={color} />
          <Badge text={`Clonazepam: ${cohort.clonazepam_responsive ? '✓ effective' : '✗ not indicated'}`} color={cohort.clonazepam_responsive ? '#2e7d32' : '#b71c1c'} />
          <Badge text={`NBS: ${cohort.nbs_detected_pct > 0.5 ? '✓ detected' : '✗ missed'}`} color={cohort.nbs_detected_pct > 0.5 ? '#1565c0' : '#757575'} />
          {cohort.lactic_acidosis_pct > 0.3 && <Badge text={`Lactic acidosis: ${Math.round(cohort.lactic_acidosis_pct * 100)}%`} color="#6a1b9a" />}
          {cohort.burst_suppression_pct > 0.3 && <Badge text={`Burst-suppression EEG: ${Math.round(cohort.burst_suppression_pct * 100)}%`} color="#1a237e" />}
          {cohort.hypertonia_pct > 0.3 && <Badge text={`Hypertonia: ${Math.round(cohort.hypertonia_pct * 100)}%`} color="#e65100" />}
          {cohort.hiccupping_pct > 0.2 && <Badge text={`Hiccupping: ${Math.round(cohort.hiccupping_pct * 100)}%`} color="#00695c" />}
          {cohort.nose_tap_pct > 0.3 && <Badge text={`Nose-tap+: ${Math.round(cohort.nose_tap_pct * 100)}%`} color="#b71c1c" />}
          <Badge text={`Onset: ~${cohort.median_onset_months}mo`} color="#37474f" />
          <Badge text={`Pathognomonic: ${cohort.pathognomonic ?? '—'}`} color="#4a148c" />
        </div>
      )}
    </div>
  );
}

export default function HeredNKHGlycineEncephalopathyHyperekplexiaAtlasPage() {
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
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center' }}>Loading NKH / Glycine Encephalopathy / Hyperekplexia Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const cohortMap = {};
  (overview?.gene_cohorts ?? []).forEach(c => { cohortMap[c.gene] = c; });

  return (
    <div style={{ fontFamily: 'sans-serif', maxWidth: 1100, margin: '0 auto', padding: '20px 16px' }}>
      <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
        🧬 Hereditary-NKH-Glycine-Encephalopathy-Hyperekplexia-Atlas
      </h1>
      <p style={{ fontSize: 13, color: '#555', marginBottom: 4 }}>
        Complete 8-Gene Reference · GLDC · AMT · GCSH · DLD · GLRA1 · GLRB · SLC6A5 · SLC6A9
      </p>
      <p style={{ fontSize: 12, color: '#777', marginBottom: 16 }}>
        Seeds {overview?.seeds} · {overview?.total_patients} patients · Non-Ketotic Hyperglycinemia (NKH) + Glycine Receptor/Transporter Disorders
      </p>

      {/* DDx Banner */}
      <div style={{ background: '#e8eaf6', border: '2px solid #3949ab', borderRadius: 8, padding: '10px 16px', marginBottom: 16, fontSize: 13 }}>
        <strong>⚡ Primary DDx Split:</strong>{' '}
        {overview?.atlas_summary?.critical_ddx_rule ?? 'NKH (CSF:plasma glycine >0.08 + GCS enzyme deficient) vs Hyperekplexia (glycine NORMAL + startle disease)'}
        <div style={{ marginTop: 6, fontSize: 12, color: '#555' }}>
          <strong>NKH group:</strong> {(overview?.atlas_summary?.nkh_group_genes ?? []).join(', ')} — glycine elevated CSF+plasma, burst-suppression, hypotonia, sodium benzoate+dextromethorphan&nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>Hyperekplexia group:</strong> {(overview?.atlas_summary?.hyperekplexia_group_genes ?? []).join(', ')} — glycine NORMAL, hypertonia, startle, clonazepam highly effective&nbsp;&nbsp;|&nbsp;&nbsp;
          <strong>Mimics NKH but GCS normal:</strong> {overview?.atlas_summary?.gcs_enzyme_normal_mimics_nkh ?? 'SLC6A9'}
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 20, border: 'none', cursor: 'pointer', fontSize: 13,
            background: tab === t ? '#1a237e' : '#e8eaf6', color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'Overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Total Patients', value: overview?.total_patients },
              { label: 'Genes', value: overview?.genes_count },
              { label: 'Seeds', value: overview?.seeds },
              { label: 'NKH Genes', value: (overview?.atlas_summary?.nkh_group_genes ?? []).length },
              { label: 'Hyperekplexia Genes', value: (overview?.atlas_summary?.hyperekplexia_group_genes ?? []).length },
              { label: 'GCS-Normal Mimic', value: overview?.atlas_summary?.gcs_enzyme_normal_mimics_nkh ?? 'SLC6A9' },
            ].map(({ label, value }) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: '10px 14px', textAlign: 'center' }}>
                <div style={{ fontSize: 20, fontWeight: 700 }}>{value}</div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 2 }}>{label}</div>
              </div>
            ))}
          </div>
          <h3 style={{ fontSize: 16, marginBottom: 10 }}>All 8 Genes</h3>
          {Object.entries(GENE_COLORS).map(([g, col]) => (
            <GeneCard key={g} gene={g} color={col} info={GENE_INFO[g]} cohort={cohortMap[g]} />
          ))}
        </div>
      )}

      {/* ── Gene Table Tab ── */}
      {tab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Gene</th>
                <th style={{ padding: '8px 10px' }}>Group</th>
                <th style={{ padding: '8px 10px' }}>CSF:Plasma Ratio</th>
                <th style={{ padding: '8px 10px' }}>NBS</th>
                <th style={{ padding: '8px 10px' }}>Clonazepam</th>
                <th style={{ padding: '8px 10px' }}>Lactic Acidosis</th>
                <th style={{ padding: '8px 10px' }}>First-line Rx</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Key Distinguisher</th>
              </tr>
            </thead>
            <tbody>
              {PHENOTYPE_ROWS.map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                  <td style={{ padding: '8px 10px' }}>{row.group}</td>
                  <td style={{ padding: '8px 10px', fontWeight: 700, color: row.glycine_ratio === 'NORMAL' ? '#2e7d32' : row.glycine_ratio.startsWith('>') ? '#b71c1c' : '#e65100', textAlign: 'center' }}>{row.glycine_ratio}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center' }}>{row.nbs}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', color: row.clonaz === '✓' ? '#2e7d32' : '#888', fontWeight: row.clonaz === '✓' ? 700 : 400 }}>{row.clonaz}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center', color: row.la === '↑↑' ? '#b71c1c' : '#888', fontWeight: row.la === '↑↑' ? 700 : 400 }}>{row.la}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{row.rx}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{row.unique}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Pathway Map Tab ── */}
      {tab === 'Pathway Map' && (
        <div>
          {Object.entries(breakdown?.pathway_groups ?? {}).map(([key, group]) => (
            <div key={key} style={{ marginBottom: 20, border: '1px solid #ddd', borderRadius: 8, padding: 16 }}>
              <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8, color: key.includes('nkh') ? '#1a237e' : '#b71c1c' }}>
                {key.includes('nkh') ? '🔵 NKH / Glycine Cleavage System (GCS)' : '🔴 Hyperekplexia / Glycine Receptor & Transporter'}
              </h3>
              <div style={{ marginBottom: 8 }}><strong>Genes:</strong> {(group.genes ?? []).join(', ')}</div>
              <div style={{ marginBottom: 6, fontSize: 13 }}><strong>Mechanism:</strong> {group.mechanism}</div>
              <div style={{ marginBottom: 6, fontSize: 13 }}><strong>Shared Feature:</strong> {group.shared_feature}</div>
              <div style={{ fontSize: 13, color: '#555' }}><strong>Key DDx:</strong> {group.key_ddx}</div>
            </div>
          ))}

          <h3 style={{ fontSize: 15, marginBottom: 10 }}>10-Step Glycine Diagnostic Cascade</h3>
          <ol style={{ paddingLeft: 20, fontSize: 13 }}>
            {(breakdown?.glycine_diagnostic_cascade ?? []).map((step, i) => (
              <li key={i} style={{ marginBottom: 8, lineHeight: 1.5 }}>{step}</li>
            ))}
          </ol>

          {breakdown?.vigabatrin_valproate_warning && (
            <div style={{ background: '#fce4ec', border: '2px solid #c62828', borderRadius: 8, padding: 12, marginTop: 16, fontSize: 13 }}>
              <strong>⚠️ Drug Warning (NKH):</strong> {breakdown.vigabatrin_valproate_warning}
            </div>
          )}
          {breakdown?.clonazepam_nkh_distinction && (
            <div style={{ background: '#e8f5e9', border: '2px solid #2e7d32', borderRadius: 8, padding: 12, marginTop: 10, fontSize: 13 }}>
              <strong>✅ Clonazepam vs NKH Treatment:</strong> {breakdown.clonazepam_nkh_distinction}
            </div>
          )}
        </div>
      )}

      {/* ── Treatment Tab ── */}
      {tab === 'Treatment' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(300px,1fr))', gap: 16, marginBottom: 20 }}>
            {(breakdown?.treatment_pathways ?? []).map(tp => (
              <div key={tp.gene} style={{ border: `2px solid ${GENE_COLORS[tp.gene] || '#999'}`, borderRadius: 8, padding: 14 }}>
                <div style={{ fontWeight: 700, color: GENE_COLORS[tp.gene], marginBottom: 6, fontSize: 14 }}>
                  {tp.gene} — {tp.disease}
                </div>
                <div style={{ fontSize: 12, marginBottom: 4 }}><strong>First-line:</strong> {tp.first_line}</div>
                <div style={{ fontSize: 12, marginBottom: 4 }}><strong>Precision:</strong> {tp.precision_therapy}</div>
                <div style={{ fontSize: 12, marginBottom: 4, color: '#b71c1c' }}><strong>⚠ Avoid:</strong> {tp.avoid}</div>
                <div style={{ fontSize: 12, color: '#555' }}><strong>Monitor:</strong> {tp.monitoring}</div>
              </div>
            ))}
          </div>

          <h3 style={{ fontSize: 15, marginBottom: 10 }}>Treatment Matrix</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#37474f', color: '#fff' }}>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Treatment</th>
                <th style={{ padding: '8px 10px' }}>Mechanism</th>
                <th style={{ padding: '8px 10px' }}>Target Genes</th>
                <th style={{ padding: '8px 10px', textAlign: 'left' }}>Notes</th>
              </tr>
            </thead>
            <tbody>
              {(definitions?.treatment_matrix ?? []).map((tm, i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: '8px 10px', fontWeight: 700 }}>{tm.treatment}</td>
                  <td style={{ padding: '8px 10px' }}>{tm.mechanism}</td>
                  <td style={{ padding: '8px 10px', textAlign: 'center' }}>{(tm.target_genes ?? []).join(', ')}</td>
                  <td style={{ padding: '8px 10px', fontSize: 11 }}>{tm.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'Definitions' && (
        <div>
          <h3 style={{ fontSize: 15, marginBottom: 10 }}>Key Distinguishers</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))', gap: 12, marginBottom: 20 }}>
            {(definitions?.key_distinguishers ?? []).map((kd, i) => (
              <div key={i} style={{ background: '#f3f4f6', borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{kd.feature ?? kd.title ?? kd.distinguisher}</div>
                <div style={{ fontSize: 12, color: '#333' }}>{kd.detail ?? kd.description ?? kd.clinical_significance}</div>
              </div>
            ))}
          </div>

          <h3 style={{ fontSize: 15, marginBottom: 10 }}>Glossary</h3>
          {(definitions?.definitions ?? []).map((def, i) => (
            <div key={i} style={{ marginBottom: 12, padding: '10px 14px', background: i % 2 === 0 ? '#f8f9fa' : '#fff', borderRadius: 6, border: '1px solid #e0e0e0' }}>
              <span style={{ fontWeight: 700, color: '#1a237e' }}>{def.term}</span>
              <span style={{ marginLeft: 10, fontSize: 13, color: '#444' }}>{def.definition}</span>
            </div>
          ))}
        </div>
      )}

      <div style={{ marginTop: 24, fontSize: 11, color: '#aaa', textAlign: 'center' }}>
        Hereditary-NKH-Glycine-Encephalopathy-Hyperekplexia-Atlas · Seeds {overview?.seeds} ·
        GLDC/AMT/GCSH/DLD (NKH) + GLRA1/GLRB/SLC6A5/SLC6A9 (Hyperekplexia) ·
        {overview?.total_patients} patients
      </div>
    </div>
  );
}
