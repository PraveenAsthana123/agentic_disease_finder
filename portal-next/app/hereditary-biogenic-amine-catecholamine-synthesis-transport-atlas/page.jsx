'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-biogenic-amine-catecholamine-synthesis-transport-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Treatment', 'Definitions'];

const GENE_COLORS = {
  'TH':      '#1a237e',  // deep navy       — TyrosineHydroxylase; DRD3; diurnal variation; HVA low; L-DOPA responsive
  'DDC':     '#1b5e20',  // dark green       — AADC; oculogyric crises PATHOGNOMONIC; dual substrate (DA+5HT); gene therapy
  'DBH':     '#880e4f',  // deep crimson     — DBH deficiency; NE absent; DA high; droxidopa CURATIVE
  'SLC18A2': '#4a148c',  // deep purple      — VMAT2; all monoamines depleted; infantile Parkinsonism-dystonia
  'SLC6A3':  '#e65100',  // burnt orange     — DAT1; HVA paradoxically HIGH; DAT SPECT absent
  'MAOA':    '#b71c1c',  // dark red         — Brunner syndrome; XLR; aggression PATHOGNOMONIC; monoamine accumulation
  'SPR':     '#004d40',  // dark teal        — sepiapterin reductase; BH4 distal; biopterin ELEVATED; BOTH DA+5HT low
  'GCH1':    '#37474f',  // dark slate       — GCH1 Segawa DRD1; AD-LOF; diurnal variation; dramatic L-DOPA; neopterin+biopterin LOW
};

const GENE_INFO = {
  'TH':      { full: 'TH-528aa / AR / Tyrosine-Hydroxylase / Rate-limiting-dopamine-norepinephrine-synthesis / 11p15.5 / TH-Deficiency-DRD3 / HVA-LOW-5HIAA-NORMAL / DIURNAL-VARIATION-PATHOGNOMONIC-mild / L-DOPA-responsive-mild / Spectrum-severe-encephalopathy-to-mild-DRD3', locus: '11p15.5', size: '528 aa / 45 kDa', inh: 'AR; BH4-dependent; rate-limiting step dopamine/NE synthesis; SPECTRUM from severe (progressive encephalopathy, L-DOPA-nonresponsive, TH0) to mild (DOPA-responsive dystonia type 3, diurnal variation); CSF: HVA LOW + MHPG LOW + 5-HIAA NORMAL (distinguishes from DDC/AADC where 5-HIAA also low)' },
  'DDC':     { full: 'DDC-480aa / AR / DOPA-Decarboxylase-AADC / Decarboxylates-L-DOPA-to-dopamine-AND-5-HTP-to-serotonin / 7p12.2 / AADC-Deficiency / OCULOGYRIC-CRISES-PATHOGNOMONIC / HVA-LOW-5HIAA-LOW-L-DOPA-ELEVATED-5HTP-ELEVATED / DUAL-substrate-dopamine-serotonin / AAV2-DDC-gene-therapy', locus: '7p12.2', size: '480 aa / 54 kDa', inh: 'AR; DUAL substrate: L-DOPA→dopamine AND 5-HTP→serotonin; OCULOGYRIC CRISES pathognomonic (involuntary upward gaze, extreme distress, last minutes-hours); CSF: HVA LOW + 5-HIAA LOW + L-DOPA ELEVATED + 5-HTP ELEVATED (substrates accumulate = pathognomonic CSF signature); plasma AADC enzyme activity confirmatory; gene therapy (AAV2-DDC intraputaminal) clinically transformative' },
  'DBH':     { full: 'DBH-617aa / AR / Dopamine-Beta-Hydroxylase / Converts-dopamine-to-norepinephrine / 9q33.2 / DBH-Deficiency / NE-ABSENT-DA-HIGH-PATHOGNOMONIC / Severe-orthostatic-hypotension / Undetectable-serum-DBH-activity / L-DOPS-droxidopa-CURATIVE', locus: '9q33.2', size: '617 aa / 75 kDa', inh: 'AR; converts dopamine→norepinephrine in chromaffin granules; NOREPINEPHRINE ABSENT + DOPAMINE MARKEDLY ELEVATED pathognomonic (serum + urine catecholamines); severe orthostatic hypotension, ptosis, nasal stuffiness, retrograde ejaculation (males); undetectable serum DBH enzyme activity; L-DOPS (droxidopa) = synthetic NE precursor, CURATIVE, FDA-approved' },
  'SLC18A2': { full: 'SLC18A2-514aa / AR / VMAT2-Vesicular-Monoamine-Transporter-2 / Packages-all-monoamines-into-vesicles / 10q25.3 / DTDS-VMAT2-Infantile-Parkinsonism-Dystonia / ALL-MONOAMINES-DEPLETED-PATHOGNOMONIC / Dopamine-SPECT-reduced / Infantile-hypotonia-to-hypertonia-parkinsonism', locus: '10q25.3', size: '514 aa / 55 kDa', inh: 'AR; packages ALL monoamines (dopamine, serotonin, NE, epinephrine) into secretory vesicles; ALL MONOAMINES depleted → pathognomonic; dopamine SPECT reduced presynaptic uptake (like PD in infant); CSF: HVA LOW + 5-HIAA LOW + HVA/5-HIAA ratio maintained (all depleted); treatment partial: L-DOPA + MAO-B inhibitor + dopamine agonists; poor long-term prognosis' },
  'SLC6A3':  { full: 'SLC6A3-620aa / AR / DAT1-Dopamine-Transporter / Reuptakes-dopamine-from-synapse / 5p15.33 / DAT-Deficiency-DTDS-Type1 / HVA-PARADOXICALLY-ELEVATED-PATHOGNOMONIC / DAT-SPECT-absent / Infantile-Parkinsonism-hyperkinetic / Dopamine-excess-synapse-postsynaptic-downregulation-paradox', locus: '5p15.33', size: '620 aa / 69 kDa', inh: 'AR; failure to reuptake dopamine from synapse → dopamine excess → postsynaptic downregulation → net parkinsonism-dystonia PARADOX; CSF HVA PARADOXICALLY ELEVATED (excess dopamine turnover) despite clinical parkinsonism PATHOGNOMONIC; DAT SPECT (123I-ioflupane): absent DAT binding = diagnostic; neonatal hypotonia → infantile parkinsonism + hyperkinetic movement' },
  'MAOA':    { full: 'MAOA-527aa / XLR / Monoamine-Oxidase-A / Degrades-serotonin-dopamine-norepinephrine / Xp11.3 / Brunner-Syndrome / MALES-ONLY / IMPULSIVE-AGGRESSIVE-BEHAVIOR-PATHOGNOMONIC / Low-IQ / Elevated-urinary-monoamine-metabolites / Carriers-asymptomatic', locus: 'Xp11.3', size: '527 aa / 60 kDa', inh: 'XLR (X-linked recessive); MALES ONLY (hemizygous); LOF → serotonin + dopamine + norepinephrine accumulate; borderline IQ + IMPULSIVE AGGRESSIVE BEHAVIOR pathognomonic in males; elevated urinary serotonin (5-HIAA), HVA, VMA, normetanephrine; female carriers asymptomatic (heterozygous); behavioral management primary; SSRIs sometimes used; MAO inhibitors would worsen' },
  'SPR':     { full: 'SPR-261aa / AR / Sepiapterin-Reductase / Last-step-BH4-de-novo-synthesis / 2p13.2 / Sepiapterin-Reductase-Deficiency / BH4-cofactor-TH-TPH-NOS / BIOPTERIN-ELEVATED-SEPIAPTERIN-ELEVATED-PATHOGNOMONIC / BOTH-HVA-AND-5HIAA-LOW / Normal-phenylalanine / Oral-BH4-NOT-crossing-BBB', locus: '2p13.2', size: '261 aa / 28 kDa', inh: 'AR; last step BH4 de novo synthesis in peripheral tissues; BH4 is cofactor for TH (dopamine), TPH (serotonin), NOS (nitric oxide); DISTINCTIVE CSF: HVA LOW + 5-HIAA LOW + BIOPTERIN ELEVATED + SEPIAPTERIN elevated (pathognomonic); BOTH dopamine AND serotonin low (unlike TH where only dopamine/NE affected); NORMAL phenylalanine (contrast with PTS/QDPR); oral BH4 does NOT cross BBB adequately; treatment: L-DOPA/carbidopa + 5-HTP mandatory' },
  'GCH1':    { full: 'GCH1-250aa / AD-LOF / GTP-Cyclohydrolase-I / FIRST-step-BH4-synthesis / 14q22.2 / Segawa-Disease-DRD1 / DIURNAL-VARIATION-PATHOGNOMONIC / DRAMATIC-LOW-DOSE-L-DOPA-RESPONSE-PATHOGNOMONIC / AD-haploinsufficiency-female-4:1-penetrance / Neopterin-biopterin-LOW-CSF / Normal-phenylalanine', locus: '14q22.2', size: '250 aa / 28 kDa', inh: 'AD-LOF (Segawa = haploinsufficiency, female:male 4:1 penetrance); AR complete LOF = severe neonatal; GCH1 = FIRST step BH4 biosynthesis; BH4 rate-limiting cofactor for TH; DIURNAL VARIATION pathognomonic (worse by evening, better after sleep); DRAMATIC response at VERY LOW dose L-DOPA (1-2 mg/kg/day) distinguishes from PD (needs 10-fold higher) and DYT1 (L-DOPA ineffective); CSF: HVA LOW + 5-HIAA LOW + neopterin LOW + biopterin LOW (all decreased, contrast with SPR: biopterin elevated); lifelong L-DOPA; no tolerance' },
};

const PHENOTYPE_ROWS = [
  { gene: 'TH',      group: 'Dopamine Synthesis — TH Deficiency / DRD3',      csf: 'HVA↓, 5-HIAA nl', pathway: 'DA synthesis', xl: '✗', pathognomonic: 'Diurnal variation (mild); L-DOPA responsive; 5-HIAA NORMAL', rx: 'L-DOPA/carbidopa (responsive in mild)',  unique: 'SPECTRUM: severe (TH0 — progressive encephalopathy, L-DOPA-non-responsive) to mild (DRD3 — diurnal dystonia, L-DOPA responsive); CSF 5-HIAA NORMAL is key DDx from AADC (where serotonin ALSO deficient); diurnal variation shared with GCH1; BH4-dependent enzyme' },
  { gene: 'DDC',     group: 'Dopamine + Serotonin — AADC Deficiency',          csf: 'HVA↓, 5HIAA↓, L-DOPA↑, 5HTP↑', pathway: 'DA+5HT synthesis', xl: '✗', pathognomonic: 'Oculogyric crises PATHOGNOMONIC; substrates accumulate', rx: 'DA agonist + MAO-B inhibitor + pyridoxine; gene therapy', unique: 'DUAL substrate enzyme (L-DOPA→DA AND 5-HTP→5HT); oculogyric crises pathognomonic (involuntary upward gaze, extreme distress); CSF signature: HVA+5-HIAA LOW + L-DOPA+5-HTP ELEVATED; plasma AADC enzyme activity diagnostic; AAV2-DDC intraputaminal gene therapy clinically transformative' },
  { gene: 'DBH',     group: 'Norepinephrine Synthesis — DBH Deficiency',        csf: 'NE absent, DA↑↑', pathway: 'NE synthesis', xl: '✗', pathognomonic: 'NE absent + DA markedly high PATHOGNOMONIC; droxidopa CURATIVE', rx: 'L-DOPS (droxidopa) — CURATIVE, FDA-approved', unique: 'NOREPINEPHRINE ABSENT in serum/urine; dopamine markedly elevated (NE/DA ratio → 0); severe orthostatic hypotension (syncope, falls), ptosis, nasal congestion, retrograde ejaculation; undetectable serum DBH enzyme activity; droxidopa (synthetic NE precursor) is CURATIVE — rare case of truly curative pharmacotherapy in movement disorder' },
  { gene: 'SLC18A2', group: 'Vesicular Transport — VMAT2 Deficiency',          csf: 'HVA↓, 5HIAA↓', pathway: 'Vesicular packaging', xl: '✗', pathognomonic: 'ALL monoamines depleted; DOPAMINE SPECT reduced', rx: 'L-DOPA + MAO-B inhibitor + DA agonists (partial)', unique: 'ALL monoamines depleted (dopamine + serotonin + NE + epinephrine — all vesicular packaging fails); dopamine SPECT reduced presynaptic uptake (like PD in infant); CSF HVA + 5-HIAA both low (HVA/5-HIAA ratio maintained, unlike DDC where both metabolites low but substrate patterns differ); poor long-term prognosis despite treatment' },
  { gene: 'SLC6A3',  group: 'Dopamine Reuptake — DAT Deficiency',              csf: 'HVA↑↑ PARADOX', pathway: 'DA reuptake', xl: '✗', pathognomonic: 'HVA PARADOXICALLY HIGH despite parkinsonism; DAT SPECT absent', rx: 'L-DOPA partial; DA agonists; gene therapy investigational', unique: 'PARADOX: dopamine reuptake failure → synaptic dopamine excess → postsynaptic D1/D2 downregulation → net parkinsonism-dystonia despite ELEVATED dopamine; CSF HVA ELEVATED (excess DA turnover) is PATHOGNOMONIC in context of parkinsonism; DAT SPECT (123I-ioflupane) absent binding is diagnostic; management very challenging' },
  { gene: 'MAOA',    group: 'Monoamine Degradation — Brunner Syndrome',         csf: 'Monoamines↑', pathway: 'MAO degradation', xl: '✓ XLR', pathognomonic: 'XLR males; impulsive aggression PATHOGNOMONIC', rx: 'Behavioral management; SSRIs (caution); no MAO inhibitors', unique: 'X-LINKED RECESSIVE — males only (hemizygous); LOF → serotonin + dopamine + NE accumulate; borderline IQ + impulsive aggressive behavior pathognomonic combination; elevated urine 5-HIAA, HVA, VMA, normetanephrine; female carriers asymptomatic (heterozygous expression sufficient); original gene mapped by Brunner 1993 (Science); MAO inhibitors ABSOLUTELY contraindicated' },
  { gene: 'SPR',     group: 'BH4 Synthesis (Distal) — SPR Deficiency',         csf: 'HVA↓, 5HIAA↓, biopterin↑', pathway: 'BH4 synthesis', xl: '✗', pathognomonic: 'BIOPTERIN ELEVATED in CSF (DDx GCH1 where biopterin low)', rx: 'L-DOPA/carbidopa + 5-HTP (BOTH required); oral BH4 NOT crossing BBB', unique: 'DISTINCTIVE: biopterin ELEVATED in CSF (sepiapterin accumulates proximal to block, converted to biopterin); this distinguishes SPR from GCH1 (where neopterin + biopterin are BOTH LOW); BOTH HVA AND 5-HIAA low (dopamine + serotonin affected — unlike TH); normal phenylalanine (contrast with other BH4 disorders); oral BH4 does NOT penetrate CNS adequately — must treat empirically with L-DOPA + 5-HTP' },
  { gene: 'GCH1',    group: 'BH4 Synthesis (Proximal) — Segawa DRD1',          csf: 'HVA↓, 5HIAA↓, neopterin↓, biopterin↓', pathway: 'BH4 synthesis', xl: '✗', pathognomonic: 'Diurnal variation; DRAMATIC low-dose L-DOPA response PATHOGNOMONIC', rx: 'L-DOPA/carbidopa (very low dose, lifelong, no tolerance)', unique: 'AD haploinsufficiency = Segawa disease (female:male 4:1, diurnal variation); DRAMATIC low-dose L-DOPA (1-2 mg/kg/day) is pathognomonic response — PD needs 10x higher dose, DYT1 does not respond at all; CSF: HVA + 5-HIAA + neopterin + BIOPTERIN all LOW (all pre-SPR pathway depleted — contrast SPR: biopterin elevated); normal phenylalanine in AD form; AR complete LOF: severe neonatal hyperphen + neurotransmitter deficiency' },
];

const SEED_BASE_DISPLAY = {
  'TH': 4230, 'DDC': 4231, 'DBH': 4232, 'SLC18A2': 4233,
  'SLC6A3': 4234, 'MAOA': 4235, 'SPR': 4236, 'GCH1': 4237,
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color || '#333', color: '#fff', padding: '2px 7px',
      borderRadius: 4, fontSize: 11, marginRight: 4, marginBottom: 4, display: 'inline-block'
    }}>{text}</span>
  );
}

function GeneCard({ gene, color, info, geneData, cohortData }) {
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 12, marginBottom: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <div>
          <span style={{ background: color, color: '#fff', fontWeight: 700, padding: '3px 10px', borderRadius: 5, fontSize: 15 }}>{gene}</span>
          <span style={{ marginLeft: 10, fontSize: 12, color: '#555' }}>{geneData?.disease ?? ''}</span>
        </div>
        <span style={{ fontSize: 11, color: '#888' }}>seed {SEED_BASE_DISPLAY[gene] ?? '—'} · n=40</span>
      </div>
      <div style={{ marginTop: 8, fontSize: 12, color: '#333' }}>{info?.full}</div>
      <div style={{ marginTop: 6, fontSize: 12 }}>
        <strong>Locus:</strong> {info?.locus} &nbsp;|&nbsp;
        <strong>Size:</strong> {info?.size} &nbsp;|&nbsp;
        <strong>Inheritance:</strong> {info?.inh?.split(';')[0]}
      </div>
      {geneData && (
        <div style={{ marginTop: 6, fontSize: 12, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text={`Group: ${geneData.pathway_group ?? '—'}`} color={color} />
          <Badge text={`Key CSF: ${geneData.key_csf_pattern?.slice(0, 40) ?? '—'}`} color="#1565c0" />
          {geneData.has_oculogyric_crises && <Badge text="OCULOGYRIC CRISES PATHOGNOMONIC" color="#1b5e20" />}
          {geneData.has_orthostatic_hypotension && <Badge text="Orthostatic Hypotension" color="#880e4f" />}
          {geneData.has_diurnal_variation && <Badge text="Diurnal Variation" color="#1a237e" />}
          {geneData.has_dramatic_ldopa_response && <Badge text="DRAMATIC L-DOPA response" color="#37474f" />}
          {geneData.is_x_linked && <Badge text="X-Linked" color="#b71c1c" />}
          {geneData.csf_biopterin_elevated && <Badge text="Biopterin ELEVATED" color="#004d40" />}
          {geneData.csf_hva_paradox && <Badge text="HVA PARADOX (elevated in parkinsonism)" color="#e65100" />}
          {geneData.all_monoamines_depleted && <Badge text="ALL Monoamines depleted" color="#4a148c" />}
          {geneData.ne_absent && <Badge text="NE ABSENT" color="#880e4f" />}
          {geneData.gene_therapy_available && <Badge text="Gene Therapy Available" color="#1b5e20" />}
          {cohortData && <Badge text={`Severe: ${Math.round((cohortData.pct_severe ?? 0) * 100)}%`} color={cohortData.pct_severe > 0.4 ? '#b71c1c' : '#388e3c'} />}
          {geneData.pathognomonic_note && (
            <div style={{ width: '100%', marginTop: 4, background: '#f5f5f5', padding: '4px 8px', borderRadius: 4, fontSize: 11, color: '#333' }}>
              <strong>⚡ Pathognomonic:</strong> {geneData.pathognomonic_note?.split(';')[0]}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default function HeredBiogenicAmineAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const base = `${API}/api/${SLUG}`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov);
      setBreakdown(bk);
      setDefinitions(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, textAlign: 'center', fontFamily: 'monospace' }}>Loading Biogenic Amine Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: '#b71c1c', fontFamily: 'monospace' }}>Error: {error}</div>;

  const genes = overview?.genes ?? Object.keys(GENE_COLORS);
  const geneTable = overview?.gene_table ?? {};
  const cascade = overview?.primary_cascade_split ?? '';

  return (
    <div style={{ fontFamily: 'monospace', maxWidth: 1200, margin: '0 auto', padding: '16px 12px' }}>
      <div style={{ background: '#1a237e', color: '#fff', borderRadius: 8, padding: '14px 20px', marginBottom: 16 }}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 700 }}>
          🧬 Hereditary Biogenic Amine / Catecholamine Synthesis &amp; Transport Atlas
        </h1>
        <div style={{ fontSize: 12, marginTop: 6, opacity: 0.85 }}>
          Complete 8-Gene Reference · TH · DDC · DBH · SLC18A2 · SLC6A3 · MAOA · SPR · GCH1
          &nbsp;·&nbsp; 320 Patients (8×40, seeds 4230–4237)
          &nbsp;·&nbsp; Dopamine · Serotonin · Norepinephrine · BH4 Pathway
        </div>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 16, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '6px 16px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === t ? '#1a237e' : '#e8eaf6', color: tab === t ? '#fff' : '#1a237e',
            fontFamily: 'monospace', fontWeight: tab === t ? 700 : 400, fontSize: 13
          }}>{t}</button>
        ))}
      </div>

      {/* ── Overview ── */}
      {tab === 'Overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 10, marginBottom: 16 }}>
            {[
              { label: 'Total Patients', val: overview?.total_patients ?? 320 },
              { label: 'Genes', val: '8 (TH / DDC / DBH / SLC18A2 / SLC6A3 / MAOA / SPR / GCH1)' },
              { label: 'Seeds', val: overview?.seeds ?? '4230–4237' },
              { label: 'Oculogyric Crises Gene', val: (overview?.oculogyric_crises_genes ?? []).join(', ') || 'DDC' },
              { label: 'Diurnal Variation Genes', val: (overview?.diurnal_variation_genes ?? []).join(', ') || 'TH, GCH1' },
              { label: 'NE-Absent Gene', val: overview?.ne_absent_gene ?? 'DBH' },
              { label: 'HVA Paradox Gene', val: overview?.csf_hva_paradox_gene ?? 'SLC6A3' },
              { label: 'Gene Therapy Gene', val: overview?.gene_therapy_gene ?? 'DDC' },
            ].map(({ label, val }) => (
              <div key={label} style={{ background: '#e8eaf6', borderRadius: 6, padding: '10px 14px' }}>
                <div style={{ fontSize: 11, color: '#555', textTransform: 'uppercase', letterSpacing: 0.5 }}>{label}</div>
                <div style={{ fontSize: 14, fontWeight: 700, color: '#1a237e', marginTop: 2 }}>{val}</div>
              </div>
            ))}
          </div>

          <div style={{ background: '#fff3e0', borderLeft: '4px solid #e65100', borderRadius: 6, padding: '10px 14px', marginBottom: 16 }}>
            <strong>⚡ Primary Cascade Split (CSF Neurotransmitter Algorithm)</strong>
            <div style={{ fontSize: 12, marginTop: 6, whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{cascade}</div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 10 }}>
            {genes.map(g => (
              <GeneCard key={g} gene={g} color={GENE_COLORS[g] || '#333'}
                info={GENE_INFO[g]} geneData={geneTable[g]}
                cohortData={geneTable[g]?.cohort_stats} />
            ))}
          </div>
        </div>
      )}

      {/* ── Gene Table ── */}
      {tab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Disease', 'Locus', 'Size', 'Inh', 'Key CSF Pattern', 'Pathognomonic Feature', 'Treatment'].map(h => (
                  <th key={h} style={{ padding: '7px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {PHENOTYPE_ROWS.map((row, i) => (
                <tr key={row.gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff', borderBottom: '1px solid #e0e0e0' }}>
                  <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                  <td style={{ padding: '6px 10px' }}>{row.group}</td>
                  <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{GENE_INFO[row.gene]?.locus}</td>
                  <td style={{ padding: '6px 10px', whiteSpace: 'nowrap' }}>{GENE_INFO[row.gene]?.size}</td>
                  <td style={{ padding: '6px 10px' }}>{row.xl === '✓ XLR' ? '⚠ XLR' : 'AR/AD'}</td>
                  <td style={{ padding: '6px 10px', maxWidth: 180 }}>{row.csf}</td>
                  <td style={{ padding: '6px 10px', maxWidth: 220, fontSize: 11 }}>{row.pathognomonic}</td>
                  <td style={{ padding: '6px 10px', maxWidth: 180, fontSize: 11 }}>{row.rx}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Pathway Map ── */}
      {tab === 'Pathway Map' && (
        <div>
          <h3 style={{ marginTop: 0, color: '#1a237e' }}>Biogenic Amine Pathway Map</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 12, marginBottom: 18 }}>
            {[
              { group: 'BH4 Synthesis (Proximal)', genes: overview?.bh4_synthesis_group ?? ['GCH1', 'SPR'], color: '#37474f', desc: 'GCH1 (GTP cyclohydrolase I, FIRST step) → neopterin → biopterin → BH4. SPR = last step. BH4 is the rate-limiting cofactor for TH, TPH, and NOS.' },
              { group: 'Dopamine + Serotonin Synthesis', genes: overview?.synthesis_group ?? ['TH', 'DDC', 'DBH'], color: '#1b5e20', desc: 'TH: Tyrosine→L-DOPA (BH4-dependent; rate-limiting for dopamine). DDC/AADC: L-DOPA→Dopamine AND 5-HTP→Serotonin (dual substrate). DBH: Dopamine→Norepinephrine (in chromaffin granules).' },
              { group: 'Vesicular Packaging + Transport', genes: overview?.transport_group ?? ['SLC18A2', 'SLC6A3'], color: '#4a148c', desc: 'SLC18A2 (VMAT2): packages ALL monoamines into vesicles for release. SLC6A3 (DAT): reuptakes dopamine from synapse back into presynaptic terminal.' },
              { group: 'Monoamine Degradation', genes: overview?.degradation_group ?? ['MAOA'], color: '#b71c1c', desc: 'MAOA: degrades serotonin, dopamine, norepinephrine in mitochondria (oxidative deamination). LOF → monoamine accumulation → Brunner syndrome.' },
            ].map(({ group, genes: grpGenes, color, desc }) => (
              <div key={group} style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 12 }}>
                <div style={{ background: color, color: '#fff', borderRadius: 5, padding: '4px 10px', marginBottom: 8, fontWeight: 700, fontSize: 13 }}>{group}</div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
                  {grpGenes.map(g => (
                    <span key={g} style={{ background: GENE_COLORS[g] || color, color: '#fff', padding: '2px 8px', borderRadius: 4, fontSize: 12, fontWeight: 700 }}>{g}</span>
                  ))}
                </div>
                <div style={{ fontSize: 11, color: '#444', lineHeight: 1.5 }}>{desc}</div>
              </div>
            ))}
          </div>

          {breakdown?.csf_algorithm && (
            <div style={{ background: '#e8f5e9', border: '1px solid #4caf50', borderRadius: 6, padding: '10px 14px', marginBottom: 14 }}>
              <strong>🧪 CSF Neurotransmitter Diagnostic Algorithm</strong>
              <ol style={{ fontSize: 12, marginTop: 8, paddingLeft: 20, lineHeight: 1.8 }}>
                {breakdown.csf_algorithm.map((step, i) => <li key={i}>{step}</li>)}
              </ol>
            </div>
          )}

          {breakdown?.distinguishing_features && (
            <div>
              <h4 style={{ color: '#1a237e' }}>Key Distinguishing Features</h4>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead>
                    <tr style={{ background: '#1a237e', color: '#fff' }}>
                      <th style={{ padding: '6px 10px', textAlign: 'left' }}>Feature</th>
                      <th style={{ padding: '6px 10px', textAlign: 'left' }}>Gene(s)</th>
                      <th style={{ padding: '6px 10px', textAlign: 'left' }}>Clinical Significance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {breakdown.distinguishing_features.map((f, i) => (
                      <tr key={i} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff', borderBottom: '1px solid #e0e0e0' }}>
                        <td style={{ padding: '6px 10px', fontWeight: 700 }}>{f.feature}</td>
                        <td style={{ padding: '6px 10px', color: '#1a237e' }}>{f.genes}</td>
                        <td style={{ padding: '6px 10px', fontSize: 11 }}>{f.significance}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Treatment ── */}
      {tab === 'Treatment' && (
        <div>
          <h3 style={{ marginTop: 0, color: '#1a237e' }}>Treatment Matrix — Biogenic Amine Disorders</h3>
          {definitions?.treatment_matrix?.map((tx, i) => (
            <div key={i} style={{ border: '1px solid #e0e0e0', borderRadius: 6, padding: 12, marginBottom: 10 }}>
              <div style={{ fontWeight: 700, fontSize: 14, color: '#1a237e', marginBottom: 4 }}>{tx.intervention}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 12 }}>
                <div><strong>Indications:</strong> {tx.indications}</div>
                <div><strong>Mechanism:</strong> {tx.mechanism}</div>
                <div><strong>Dose:</strong> {tx.dose}</div>
                {tx.contraindicated && <div style={{ color: '#b71c1c' }}><strong>⚠ Contraindicated:</strong> {tx.contraindicated}</div>}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions ── */}
      {tab === 'Definitions' && (
        <div>
          <h3 style={{ marginTop: 0, color: '#1a237e' }}>Key Clinical Concepts</h3>
          {definitions?.key_distinguishers?.map((d, i) => (
            <div key={i} style={{ border: '1px solid #e0e0e0', borderRadius: 6, padding: 12, marginBottom: 10 }}>
              <div style={{ fontWeight: 700, fontSize: 13, color: '#1a237e', marginBottom: 3 }}>{d.name}</div>
              <div style={{ fontSize: 12, color: '#333', marginBottom: 4 }}>{d.definition}</div>
              {d.clinical_relevance && (
                <div style={{ fontSize: 11, background: '#e8eaf6', borderRadius: 4, padding: '4px 8px', color: '#444' }}>
                  <strong>Clinical relevance:</strong> {d.clinical_relevance}
                </div>
              )}
            </div>
          ))}
          {definitions?.primary_cascade_split && (
            <div style={{ background: '#fff3e0', borderLeft: '4px solid #e65100', borderRadius: 6, padding: '10px 14px', marginTop: 12 }}>
              <strong>Primary Cascade Split (CSF Algorithm)</strong>
              <div style={{ fontSize: 12, marginTop: 6, whiteSpace: 'pre-wrap', lineHeight: 1.7 }}>{definitions.primary_cascade_split}</div>
            </div>
          )}
        </div>
      )}

      <div style={{ marginTop: 20, fontSize: 11, color: '#888', borderTop: '1px solid #e0e0e0', paddingTop: 8 }}>
        Hereditary-Biogenic-Amine-Catecholamine-Synthesis-Transport-Atlas · 8 genes · 320 patients (8×40) · seeds 4230–4237 ·
        /api/hereditary-biogenic-amine-catecholamine-synthesis-transport-atlas/overview|breakdown|definitions · Registered 2026-10-23
      </div>
    </div>
  );
}
