'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  ANK1:   '#1565c0',  // deep blue — most common HS, ankyrin-1
  SPTB:   '#b71c1c',  // deep red — beta-spectrin HS2/HE
  SLC4A1: '#e65100',  // deep orange — band 3 HS4/SAO/dRTA
  SPTA1:  '#4a148c',  // deep purple — alpha-spectrin HE/HPP neonatal
  PKLR:   '#006064',  // dark cyan — pyruvate kinase deficiency
  G6PD:   '#2e7d32',  // deep green — G6PD X-linked enzymopathy
  PIEZO1: '#f57f17',  // amber — DHS xerocytosis GOF
  KCNN4:  '#4e342e',  // deep brown — Gardos channel DHS GOF
};

const GENE_DISEASE = {
  ANK1:   'HS1 AD Haploinsufficiency — Most Common HS 30-40% — Ankyrin-1 Vertical Linkage — Splenectomy Effective — Pre-Vaccination Mandatory — Parvovirus B19 Aplastic Crisis Emergency',
  SPTB:   'HS2 AD Partial Beta-Spectrin — Also HE + HPP Neonatal — aLELY Trap — RBC Thermolability Diagnostic — Splenectomy Pre-Vaccination',
  SLC4A1: 'HS4 AD / SAO AD Malaria-Protection / AR dRTA+HA — SAO Homozygous Lethal — Alkali Mandatory dRTA — No Haemolysis SAO Heterozygote',
  SPTA1:  'HE1 AD Elliptocytosis / HPP AR Pyropoikilocytosis — Most Severe Neonatal HA — aLELY in Trans — RBC Thermolability 45°C PATHOGNOMONIC — Early Splenectomy HPP',
  PKLR:   'PK Deficiency AR Non-Spherocytic HA — Mitapivat FDA 2022 — 2,3-DPG Right-Shift Low-Hb Tolerance — Paradoxical Reticulocytosis Post-Splenectomy — Iron Chelation',
  G6PD:   'G6PD Deficiency XL 400M Worldwide — Rasburicase ABSOLUTE CI — Primaquine/Dapsone/Nitrofurantoin CI — Assay False-Normal Acute Crisis — Neonatal Jaundice Males',
  PIEZO1: 'DHS AD GOF Dehydrated Stomatocytes — High MCHC — Pseudohyperkalaemia — SPLENECTOMY CONTRAINDICATED DVT/PE — E756del European Founder — AR LOF = Lymphatic Dysplasia',
  KCNN4:  'Gardos Channelopathy AD GOF — DHS High MCHC — Pseudohyperkalaemia — SPLENECTOMY CONTRAINDICATED DVT/PE — Senicapoc Trial — Panel Testing Distinguishes PIEZO1',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /AVOID|CI|CONTRAINDICATED|ABSOLUTE|MANDATORY|PROHIBITED|MISSES|EMERGENCY|NOT.Routine|OPPOSITE|PATHOGNOMONIC|LETHAL|FATAL/i.test(text);
  const isWarning = /WARN|MONITOR|ANNUAL|CHECK|SCREEN|SURVEILLANCE|REQUIRED|PROTOCOL|STAT|FIRST|TRIAL|PHASE|ENROL|ELIGIBLE|Preferred|RESTRICTION|SCORE|Paradoxical|RECHECK|ARTIFACT/i.test(text);
  const bg = isCI ? '#b71c1c' : isWarning ? '#e65100' : '#1565c0';
  return (
    <div style={{
      background: bg, color: '#fff', borderRadius: 6, padding: '6px 12px',
      marginBottom: 8, fontSize: 13, lineHeight: 1.4,
    }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, top_alerts, genes } = data;

  const statRows = [
    ['Total patients', s.total_patients],
    ['Mean diagnostic delay (all genes)', `${s.mean_dx_delay_months} mo`],
    ['Splenectomy performed (any gene)', `${s.splenectomy_pct}%`],
    ['Transfusion required', `${s.transfusion_required_pct}%`],
    ['Gallstones (pigment)', `${s.gallstones_pct}%`],
    ['Folate prescribed', `${s.folate_prescribed_pct}%`],
    ['Aplastic crisis (parvovirus B19)', `${s.aplastic_crisis_pct}%`],
    ['Splenomegaly on presentation', `${s.splenomegaly_pct}%`],
    // ANK1
    ['ANK1 — splenectomy performed', `${s.ank1_splenectomy_pct}%`],
    ['ANK1 — EMA flow cytometry done', `${s.ank1_ema_flow_pct}%`],
    ['ANK1 — pre-splenectomy vaccination', `${s.ank1_post_spl_vaccination_pct}%`],
    ['ANK1 — penicillin prophylaxis post-splenectomy', `${s.ank1_penicillin_pct}%`],
    ['ANK1 — aplastic crisis (parvovirus B19)', `${s.ank1_aplastic_crisis_pct}%`],
    ['ANK1 — pigment gallstones', `${s.ank1_gallstones_pct}%`],
    // SPTB
    ['SPTB — splenectomy performed', `${s.sptb_splenectomy_pct}%`],
    ['SPTB — aplastic crisis', `${s.sptb_aplastic_crisis_pct}%`],
    ['SPTB — transfusion required', `${s.sptb_transfusion_pct}%`],
    ['SPTB — pigment gallstones', `${s.sptb_gallstones_pct}%`],
    // SLC4A1
    ['SLC4A1 — splenectomy performed', `${s.slc4a1_splenectomy_pct}%`],
    ['SLC4A1 — EMA flow done', `${s.slc4a1_ema_flow_pct}%`],
    ['SLC4A1 — aplastic crisis', `${s.slc4a1_aplastic_crisis_pct}%`],
    // SPTA1
    ['SPTA1 — transfusion required (HPP severe)', `${s.spta1_transfusion_pct}%`],
    ['SPTA1 — RBC thermolability test performed', `${s.spta1_rbc_thermolability_pct}%`],
    ['SPTA1 — splenectomy performed', `${s.spta1_splenectomy_pct}%`],
    ['SPTA1 — aplastic crisis', `${s.spta1_aplastic_crisis_pct}%`],
    // PKLR
    ['PKLR — mitapivat prescribed (FDA 2022)', `${s.pklr_mitapivat_pct}%`],
    ['PKLR — splenectomy performed', `${s.pklr_splenectomy_pct}%`],
    ['PKLR — iron chelation', `${s.pklr_iron_chelation_pct}%`],
    ['PKLR — paradoxical reticulocytosis post-splenectomy', `${s.pklr_paradoxical_retic_pct}%`],
    ['PKLR — transfusion required', `${s.pklr_transfusion_pct}%`],
    ['PKLR — 2,3-DPG elevated (right-shifted O2 curve)', `${s.pklr_diphosphoglycerate_pct}%`],
    // G6PD
    ['G6PD — rasburicase given (should be 0)', `${s.g6pd_rasburicase_pct}%`],
    ['G6PD — G6PD assay performed', `${s.g6pd_assay_done_pct}%`],
    ['G6PD — neonatal jaundice', `${s.g6pd_neonatal_jaundice_pct}%`],
    ['G6PD — aplastic crisis', `${s.g6pd_aplastic_crisis_pct}%`],
    // PIEZO1
    ['PIEZO1 — high MCHC (>36 g/dL)', `${s.piezo1_high_mchc_pct}%`],
    ['PIEZO1 — pseudohyperkalaemia documented', `${s.piezo1_pseudohyperkalaemia_pct}%`],
    ['PIEZO1 — splenectomy attempted (should be 0)', `${s.piezo1_splenectomy_attempted_pct}%`],
    ['PIEZO1 — DVT/PE post-splenectomy', `${s.piezo1_dvt_pe_pct}%`],
    ['PIEZO1 — ektacytometry performed', `${s.piezo1_ektacytometry_pct}%`],
    // KCNN4
    ['KCNN4 — high MCHC (>36 g/dL)', `${s.kcnn4_high_mchc_pct}%`],
    ['KCNN4 — pseudohyperkalaemia documented', `${s.kcnn4_pseudohyperkalaemia_pct}%`],
    ['KCNN4 — splenectomy attempted (should be 0)', `${s.kcnn4_splenectomy_attempted_pct}%`],
    ['KCNN4 — DVT/PE post-splenectomy', `${s.kcnn4_dvt_pe_pct}%`],
    ['KCNN4 — senicapoc trial enrolment', `${s.kcnn4_senicapoc_pct}%`],
    ['KCNN4 — ektacytometry performed', `${s.kcnn4_ektacytometry_pct}%`],
  ];

  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 700, marginBottom: 4 }}>
        {data.title || 'Hereditary Haemolytic Anaemia Atlas'}
      </h2>
      <p style={{ color: '#555', marginBottom: 16 }}>
        {data.subtitle || '8 genes · 320 patients · seeds 1526–1533'}
      </p>

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        {(genes || []).map(g => (
          <div key={g.gene} style={{
            background: GENE_COLORS[g.gene] || '#1565c0', color: '#fff', borderRadius: 8,
            padding: '10px 16px', minWidth: 120,
          }}>
            <div style={{ fontWeight: 700, fontSize: 15 }}>{g.gene}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{g.locus} · {g.aa}</div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>{(g.inheritance || '').split('—')[0].trim()}</div>
            <div style={{ fontSize: 11, opacity: 0.9, marginTop: 4 }}>{g.n_patients} pts</div>
          </div>
        ))}
      </div>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>Cohort Statistics</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13, marginBottom: 24 }}>
        <tbody>
          {statRows.map(([label, val]) => (
            <tr key={label} style={{ borderBottom: '1px solid #eee' }}>
              <td style={{ padding: '5px 8px', color: '#333' }}>{label}</td>
              <td style={{ padding: '5px 8px', fontWeight: 600, color: '#1565c0', textAlign: 'right' }}>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginBottom: 8 }}>
        Critical Alerts ({(top_alerts || []).length})
      </h3>
      {(top_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;

  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Per-Gene Breakdown</h2>
      {breakdown.map(g => (
        <div key={g.gene} style={{
          border: `2px solid ${GENE_COLORS[g.gene] || '#1565c0'}`,
          borderRadius: 10, marginBottom: 24, overflow: 'hidden',
        }}>
          <div style={{
            background: GENE_COLORS[g.gene] || '#1565c0', color: '#fff',
            padding: '10px 16px',
          }}>
            <span style={{ fontWeight: 700, fontSize: 16 }}>{g.gene}</span>
            <span style={{ marginLeft: 12, fontSize: 13 }}>{g.protein}</span>
          </div>
          <div style={{ padding: 16 }}>
            <p style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>
              <strong>Locus:</strong> {g.locus} · <strong>Size:</strong> {g.aa} ({g.kDa}) ·
              <strong> OMIM gene:</strong> {g.omim_gene} · <strong>Disease:</strong> {g.omim_disease}
            </p>
            <p style={{ fontSize: 12, color: '#555', marginBottom: 8 }}>
              <strong>Inheritance:</strong> {g.inheritance}
            </p>
            <p style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>
              <strong>Mean onset:</strong> {g.mean_onset_years} yr ·
              <strong> Mean dx delay:</strong> {g.mean_dx_delay_months} mo ·
              <strong> M:</strong> {g.sex_distribution?.M} / <strong>F:</strong> {g.sex_distribution?.F}
            </p>
            <p style={{ fontSize: 12, color: '#666', marginBottom: 8 }}>{(g.alias || '').slice(0, 400)}…</p>
            <details style={{ marginTop: 8 }}>
              <summary style={{ cursor: 'pointer', fontSize: 12, color: '#1565c0', marginBottom: 6 }}>
                Gene-Class Mechanistic Detail
              </summary>
              <p style={{ fontSize: 12, color: '#444', marginTop: 6 }}>{g.gene_class}</p>
            </details>
            <div style={{ marginTop: 10 }}>
              <strong style={{ fontSize: 12 }}>Aetiology distribution:</strong>
              {Object.entries(g.etiology_counts || {}).map(([et, cnt]) => (
                <div key={et} style={{ fontSize: 11, color: '#555', marginTop: 2 }}>
                  {et}: <strong>{cnt}</strong>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 10 }}>
              {(g.key_alerts || []).map((a, i) => <AlertBadge key={i} text={a} />)}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const { breakdown } = data;

  const rows = breakdown.map(g => ({
    gene: g.gene,
    locus: g.locus,
    aa: g.aa,
    inh: (g.inheritance || '').split('—')[0].trim(),
    disease: GENE_DISEASE[g.gene] || '',
    pts: g.n_patients,
    onset: `${g.mean_onset_years}yr`,
  }));

  return (
    <div style={{ overflowX: 'auto' }}>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 12 }}>Clinical Atlas Summary</h2>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#1565c0', color: '#fff' }}>
            {['Gene', 'Locus', 'Size', 'Inheritance', 'Disease / Key Rule', 'Pts', 'Onset'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.gene} style={{ background: i % 2 === 0 ? '#f8f8f8' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[r.gene] || '#1565c0' }}>{r.gene}</td>
              <td style={{ padding: '7px 10px' }}>{r.locus}</td>
              <td style={{ padding: '7px 10px' }}>{r.aa}</td>
              <td style={{ padding: '7px 10px' }}>{r.inh}</td>
              <td style={{ padding: '7px 10px', fontSize: 11 }}>{r.disease}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{r.pts}</td>
              <td style={{ padding: '7px 10px', textAlign: 'center' }}>{r.onset}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3 style={{ fontSize: 15, fontWeight: 700, marginTop: 24, marginBottom: 12 }}>
        Precision Treatment &amp; Investigation Matrix
      </h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr style={{ background: '#b71c1c', color: '#fff' }}>
            {['Gene', 'AVOID / Contraindicated', 'MANDATORY Investigation / Treatment', 'Special Rule'].map(h => (
              <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[
            ['ANK1', 'Splenectomy without pre-vaccination (pneumococcal/meningococcal/Hib) · Splenectomy under age 6yr without compelling indication · Ignoring parvovirus B19 aplastic crisis as "flu"', 'EMA binding flow cytometry (sensitivity 93%) · Pre-splenectomy vaccination MANDATORY ≥2 weeks before · Penicillin prophylaxis post-splenectomy · Cholecystectomy at splenectomy if gallstones · Folate 5 mg/day · Family cascade EMA + genetic testing', 'Aplastic crisis (parvovirus B19 → absent reticulocytes + sudden Hb drop) = EMERGENCY transfusion; defer splenectomy until ≥6yr for immune development'],
            ['SPTB', 'Relying on EMA alone for HE diagnosis (EMA tests Band 3, less sensitive for isolated spectrin) · Missing HPP neonatal (misdiagnosed as AIHA or infection)', 'Blood film expert morphology (spherocytes HS2; elliptocytes HE; microspherocytes+poikilocytes HPP) · RBC thermolability 45°C for HPP · Ektacytometry if available · Genetic panel SPTB+SPTA1 · Pre-splenectomy vaccination HPP', 'SPTB + aLELY in trans = HPP (most severe HA) — always test both parents; RBC thermolability at 45°C PATHOGNOMONIC for HPP'],
            ['SLC4A1', 'SAO homozygous conception without counselling (lethal in utero) · Diagnosing haemolytic anaemia in SAO heterozygote based on ovalocytosis alone (they are NOT anaemic) · Omitting alkali therapy in AR dRTA', 'SAO: confirm heterozygous (not homozygous) status — both parents SAO genotype counselling · AR dRTA: alkali therapy (sodium bicarbonate) MANDATORY · Renal ultrasound for nephrocalcinosis · Potassium supplementation · Electrolytes monitoring', 'SAO = Ala400-408 del — only 27 bp in-frame deletion; malaria protection in heterozygotes; homozygous LETHAL — prenatal diagnosis if both parents SAO'],
            ['SPTA1', 'Missing HPP neonatal HA (can be misdiagnosed as immune HA or sepsis) · Waiting too long for splenectomy in severe HPP (transfusion-dependent neonates)', 'RBC thermolability at 45°C MANDATORY in neonatal severe HA + elliptocytes · Ektacytometry (HE/HPP pattern) · SPTA1 + aLELY allele testing (both parents) · Early splenectomy in HPP if transfusion-dependent · Genetic counselling (aLELY prevalence 20-30% African)', 'aLELY is SILENT alone but DEVASTATING in trans with HE1 allele → HPP; always check BOTH alleles of both parents when HE1 found'],
            ['PKLR', 'Transfusing based on Hb number alone without assessing symptoms (2,3-DPG right-shift: patients tolerate lower Hb) · Alarming patient after splenectomy when reticulocytes rise (it is expected) · Delaying iron chelation in transfusion-dependent PKLR', 'Mitapivat (Pyrukynd FDA 2022) — offer to all eligible adults · PK enzyme assay BEFORE and AFTER splenectomy · Ferritin 6-monthly · Iron chelation if ferritin >1000 μg/L · Splenectomy (pre-vaccination) for transfusion-dependent · Explain paradoxical reticulocytosis to patient post-splenectomy', 'Mitapivat first disease-modifying therapy FDA 2022; paradoxical reticulocytosis post-splenectomy is a SIGN OF SUCCESS; 2,3-DPG right-shift means Hb 7 g/dL may be tolerated without symptoms'],
            ['G6PD', 'Rasburicase (ABSOLUTE CI — catastrophic haemolytic crisis) · Primaquine/tafenoquine without G6PD screening · Methylene blue for metHb in G6PD (CI — use O2 + ascorbic acid) · Trusting G6PD assay result during acute crisis (FALSE NORMAL)', 'G6PD assay BEFORE prescribing rasburicase, primaquine, dapsone, nitrofurantoin · Recheck G6PD 3 months post-crisis for accurate quantitation · Neonatal jaundice screening + phototherapy in deficient males · Folate supplementation · Document in drug alerts field', 'Rasburicase ABSOLUTE CI — check G6PD in ALL patients BEFORE any rasburicase prescription; assay FALSE NORMAL during crisis (reticulocytes mask deficiency) — always retest 3 months later'],
            ['PIEZO1', 'Splenectomy (CONTRAINDICATED — HIGH DVT/PE/portal vein thrombosis risk) · Treating pseudohyperkalaemia with kayexalate or dialysis (it is an artifact) · Prescribing HS management for DHS', 'Genetic panel PIEZO1+KCNN4 to confirm DHS · Confirm pseudohyperkalaemia: plasma at 37°C immediately after sampling · Ektacytometry (DHS pattern — left-shifted dehydration curve) · EMA flow is NORMAL in DHS (unlike HS) · Folate supplementation · Avoid competitive sports during haemolytic crises', 'DHS splenectomy = FATAL ERROR; pseudohyperkalaemia is K+ leakage artifact at room temperature — confirm with 37°C plasma separation; EMA normal in DHS (distinguishes from HS)'],
            ['KCNN4', 'Splenectomy (CONTRAINDICATED — DVT/PE risk identical to PIEZO1 DHS) · Assuming PIEZO1 vs KCNN4 DHS without genetic panel · Delaying senicapoc trial referral in severe cases', 'Genetic panel KCNN4+PIEZO1 (clinically identical, require panel) · Confirm pseudohyperkalaemia artifact · Ektacytometry DHS pattern · Senicapoc compassionate use / trial enrolment for severe KCNN4 DHS · Thromboprophylaxis (LMWH/DOAC) perioperatively · Folate supplementation', 'Senicapoc (Gardos channel blocker) most relevant for KCNN4-DHS specifically — cannot use for PIEZO1 DHS (different channel); requires genetic distinction; Phase III sickle cell trial showed MCHC reduction'],
          ].map(([gene, avoid, mandatory, special], i) => (
            <tr key={gene} style={{ background: i % 2 === 0 ? '#fce4ec' : '#fff' }}>
              <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#1565c0' }}>{gene}</td>
              <td style={{ padding: '7px 10px', color: '#b71c1c', fontWeight: 600 }}>{avoid}</td>
              <td style={{ padding: '7px 10px', color: '#1b5e20', fontWeight: 600 }}>{mandatory}</td>
              <td style={{ padding: '7px 10px', color: '#555' }}>{special}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  const defs = data.clinical_definitions || data.definitions || [];
  return (
    <div>
      <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 16 }}>Clinical Definitions</h2>
      {defs.map((d, i) => (
        <div key={i} style={{
          border: '1px solid #e0e0e0', borderRadius: 8,
          marginBottom: 16, padding: 16,
        }}>
          <h3 style={{ fontSize: 14, fontWeight: 700, color: '#1565c0', marginBottom: 8 }}>
            {i + 1}. {d.term}
          </h3>
          <p style={{ fontSize: 13, color: '#444', lineHeight: 1.6 }}>{d.definition}</p>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryHaemolyticAnaemiaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-haemolytic-anaemia-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(e.message));
  }, []);

  useEffect(() => {
    if ((tab === 'Gene Table' || tab === 'Clinical Atlas') && !breakdown) {
      fetch(`${API}/api/hereditary-haemolytic-anaemia-atlas/breakdown`)
        .then(r => r.json()).then(setBreakdown).catch(e => setError(e.message));
    }
    if (tab === 'Definitions' && !definitions) {
      fetch(`${API}/api/hereditary-haemolytic-anaemia-atlas/definitions`)
        .then(r => r.json()).then(setDefinitions).catch(e => setError(e.message));
    }
  }, [tab, breakdown, definitions]);

  return (
    <div style={{ padding: '1.5rem', fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto' }}>
      <div style={{ marginBottom: 8 }}>
        <span style={{
          background: '#1565c0', color: '#fff', borderRadius: 6,
          padding: '4px 12px', fontSize: 12, fontWeight: 600,
        }}>
          Hereditary Haemolytic Anaemia Atlas
        </span>
        <span style={{ marginLeft: 10, fontSize: 12, color: '#888' }}>
          8 genes · 320 patients · seeds 1526–1533
        </span>
      </div>
      <h1 style={{ fontSize: 22, fontWeight: 800, marginBottom: 4 }}>
        Hereditary-Haemolytic-Anaemia-Atlas — Complete 8-Gene Hereditary Haemolytic Anaemia Reference
      </h1>
      <p style={{ fontSize: 13, color: '#666', marginBottom: 16 }}>
        ANK1 · SPTB · SLC4A1 · SPTA1 · PKLR · G6PD · PIEZO1 · KCNN4
        — HS Splenectomy Pre-Vaccination Mandatory, G6PD Rasburicase ABSOLUTE CI,
        DHS Splenectomy CONTRAINDICATED DVT/PE, Mitapivat FDA 2022, HPP Neonatal Severe HA
      </p>

      {error && (
        <div style={{ background: '#ffebee', border: '1px solid #ef9a9a', borderRadius: 6, padding: 12, marginBottom: 16 }}>
          Error: {error}
        </div>
      )}

      <div style={{ display: 'flex', gap: 4, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t ? '#1565c0' : '#f0f0f0',
              color: tab === t ? '#fff' : '#333',
              fontWeight: tab === t ? 700 : 400,
              fontSize: 13,
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {tab === 'Overview' && <OverviewTab data={overview} />}
      {tab === 'Gene Table' && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions' && <DefinitionsTab data={definitions} />}
    </div>
  );
}
