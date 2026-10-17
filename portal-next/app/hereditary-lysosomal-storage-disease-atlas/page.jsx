'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-lysosomal-storage-disease-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'GBA':   '#4a148c',  // deep purple     -- Gaucher; splenomegaly; ERT/SRT; Parkinson-risk
  'SMPD1': '#1a237e',  // deep navy       -- Niemann-Pick-A-B; ASMD; sea-blue-histiocytes; olipudase
  'NPC1':  '#880e4f',  // deep crimson    -- Niemann-Pick-C; VSGP; cataplexy; filipin-PATHOGNOMONIC
  'HEXA':  '#b71c1c',  // dark red        -- Tay-Sachs; cherry-red-spot; Ashkenazi; no-ERT
  'GALC':  '#1b5e20',  // dark green      -- Krabbe; psychosine; HSCT-presymptomatic-ONLY; NBS-critical
  'ARSA':  '#e65100',  // burnt orange    -- MLD; sulfatide; Libmeldy-EMA2020/FDA2024; metachromatica
  'GLB1':  '#004d40',  // dark teal       -- GM1-Morquio-B; same-gene; cherry-red-spot; keratan-sulfate
  'ASAH1': '#37474f',  // dark blue-grey  -- Farber; TRIAD-nodules-joints-hoarseness; HSCT-somatic-only
};

const GENE_INFO = {
  'GBA':   { full: 'Glucocerebrosidase / Gaucher-Type1 / splenomegaly+thrombocytopenia+bone-crises-TRIAD / ERT-imiglucerase / eliglustat-SRT-FDA2014 / Parkinson-5x-heterozygotes', locus: '1q22', size: '497 aa / 59 kDa', inh: 'AR' },
  'SMPD1': { full: 'Acid-sphingomyelinase / Niemann-Pick-A-B-ASMD / sea-blue-histiocytes-BM-PATHOGNOMONIC / olipudase-alfa-XENPOZYME-FDA2022-TypeB-ONLY', locus: '11p15.4', size: '629 aa / 70 kDa', inh: 'AR' },
  'NPC1':  { full: 'NPC1-cholesterol-trafficking / Niemann-Pick-C / VSGP-PATHOGNOMONIC / gelastic-cataplexy-PATHOGNOMONIC / filipin-staining-PATHOGNOMONIC / miglustat', locus: '18q11.2', size: '1278 aa / 142 kDa', inh: 'AR' },
  'HEXA':  { full: 'Hexosaminidase-A-alpha / Tay-Sachs-GM2 / cherry-red-spot-PATHOGNOMONIC / Ashkenazi-1in30-carrier / no-ERT-available / infantile-lethal', locus: '15q23', size: '529 aa / 60 kDa', inh: 'AR' },
  'GALC':  { full: 'Galactocerebrosidase / Krabbe-globoid-cell-leukodystrophy / psychosine-PATHOGNOMONIC / HSCT-presymptomatic-ONLY-NBS-critical / irritability-stiffness-regression-TRIAD', locus: '14q31.3', size: '669 aa / 80 kDa', inh: 'AR' },
  'ARSA':  { full: 'Arylsulfatase-A / MLD-metachromatic-leukodystrophy / sulfatide-PNS-CNS / metachromatica-nerve-biopsy-PATHOGNOMONIC / Libmeldy-EMA2020-FDA2024-presymptomatic-early-only', locus: '22q13.33', size: '507 aa / 58 kDa', inh: 'AR' },
  'GLB1':  { full: 'Beta-galactosidase-1 / GM1-gangliosidosis-AND-Morquio-B-SAME-GENE / cherry-red-spot+dysostosis-TRIAD-PATHOGNOMONIC / keratan-sulfate-Morquio-B-PATHOGNOMONIC', locus: '3p22.3', size: '677 aa / 75 kDa', inh: 'AR' },
  'ASAH1': { full: 'Acid-ceramidase / Farber-lipogranulomatosis / painful-nodules+joint-contractures+hoarseness-TRIAD-PATHOGNOMONIC / HSCT-somatic-NOT-CNS / SMA-PME-variant', locus: '8p22', size: '395 aa / 44 kDa', inh: 'AR' },
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
          <div style={{ fontWeight: 600 }}>{geneAtlas.syndrome?.split('/')[0] || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ marginBottom: 8 }}>
        <Badge text={`Age Dx: ${cohort.mean_age_dx} yr`} color={bg} />
        <Badge text={`Neuro: ${cohort.neuropathy_n}/40`} color='#6a1b9a' />
        <Badge text={`ERT/Tx: ${cohort.ert_treated_n}/40`} color='#1b5e20' />
        <Badge text={`HSCT: ${cohort.hsct_n}/40`} color='#37474f' />
      </div>
      {cohort.splenomegaly_n > 0 && (
        <div style={{ fontSize: 12, background: '#ede7f6', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🫁 <b>Splenomegaly:</b> {cohort.splenomegaly_n}/40 — Gaucher TRIAD: splenomegaly + thrombocytopenia + bone crises PATHOGNOMONIC
        </div>
      )}
      {cohort.bone_crisis_n > 0 && (
        <div style={{ fontSize: 12, background: '#e8eaf6', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🦴 <b>Bone Crisis:</b> {cohort.bone_crisis_n}/40 — Gaucher bone infarcts/AVN; Erlenmeyer-flask PATHOGNOMONIC
        </div>
      )}
      {cohort.parkinson_n > 0 && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#880e4f' }}>
          🧠 <b>Parkinson phenotype:</b> {cohort.parkinson_n}/40 — GBA heterozygotes: 5x Parkinson risk
        </div>
      )}
      {cohort.sea_blue_histiocytes_n > 0 && (
        <div style={{ fontSize: 12, background: '#e3f2fd', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#1a237e' }}>
          🔬 <b>Sea-Blue Histiocytes:</b> {cohort.sea_blue_histiocytes_n}/40 — BM biopsy PATHOGNOMONIC Niemann-Pick B
        </div>
      )}
      {cohort.lung_infiltrates_n > 0 && (
        <div style={{ fontSize: 12, background: '#e0f7fa', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          💨 <b>Lung Infiltrates:</b> {cohort.lung_infiltrates_n}/40 — SMPD1 Type B ILD/alveolar pattern
        </div>
      )}
      {cohort.vsgp_n > 0 && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#880e4f' }}>
          👁 <b>VSGP:</b> {cohort.vsgp_n}/40 — Vertical Supranuclear Gaze Palsy PATHOGNOMONIC NPC1
        </div>
      )}
      {cohort.cataplexy_n > 0 && (
        <div style={{ fontSize: 12, background: '#fff8e1', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#e65100' }}>
          😴 <b>Gelastic Cataplexy:</b> {cohort.cataplexy_n}/40 — laugh-triggered atonia PATHOGNOMONIC NPC1
        </div>
      )}
      {cohort.cherry_red_spot_n > 0 && (
        <div style={{ fontSize: 12, background: '#ffebee', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#b71c1c' }}>
          👁 <b>Cherry-Red Spot:</b> {cohort.cherry_red_spot_n}/40 — fundoscopy PATHOGNOMONIC (macula surrounded by pale ganglion cells)
        </div>
      )}
      {cohort.psychosine_n > 0 && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#1b5e20' }}>
          🧪 <b>Psychosine Elevated:</b> {cohort.psychosine_n}/40 — galactosylsphingosine PATHOGNOMONIC Krabbe — oligodendrocyte toxin
        </div>
      )}
      {cohort.libmeldy_n > 0 && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          💉 <b>Libmeldy (gene therapy):</b> {cohort.libmeldy_n}/40 — EMA2020/FDA2024 presymptomatic/early-symptomatic ONLY
        </div>
      )}
      {cohort.peripheral_neuropathy_n > 0 && (
        <div style={{ fontSize: 12, background: '#f3e5f5', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚡ <b>Peripheral Neuropathy:</b> {cohort.peripheral_neuropathy_n}/40 — ARSA/MLD: both PNS + CNS involved
        </div>
      )}
      {cohort.keratan_sulfate_n > 0 && (
        <div style={{ fontSize: 12, background: '#e0f2f1', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🧬 <b>Keratan Sulfate Urine:</b> {cohort.keratan_sulfate_n}/40 — elevated PATHOGNOMONIC Morquio B (GLB1 missense)
        </div>
      )}
      {cohort.nodules_n > 0 && (
        <div style={{ fontSize: 12, background: '#eceff1', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#37474f' }}>
          🔴 <b>Subcutaneous Nodules:</b> {cohort.nodules_n}/40 — Farber TRIAD component PATHOGNOMONIC
        </div>
      )}
      {cohort.joint_contractures_n > 0 && (
        <div style={{ fontSize: 12, background: '#eceff1', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🦾 <b>Joint Contractures:</b> {cohort.joint_contractures_n}/40 — Farber TRIAD — misdiagnosed as JIA
        </div>
      )}
      {cohort.hoarseness_n > 0 && (
        <div style={{ fontSize: 12, background: '#eceff1', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🗣 <b>Hoarseness:</b> {cohort.hoarseness_n}/40 — Farber laryngeal granulomata TRIAD PATHOGNOMONIC
        </div>
      )}
      {geneAtlas.pathognomonic && (
        <div style={{ fontSize: 12, background: '#fff9c4', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          ⚠ <b>Pathognomonic:</b> {geneAtlas.pathognomonic}
        </div>
      )}
      {geneAtlas.key_therapy && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          💊 <b>Key Therapy:</b> {geneAtlas.key_therapy}
        </div>
      )}
      <div style={{ fontSize: 12, color: '#444' }}><b>Inheritance:</b> {info?.inh}</div>
      <div style={{ fontSize: 11, color: '#666', marginTop: 4 }}>{info?.full}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <div>Loading...</div>;
  const summary = data.per_gene_summary || {};
  return (
    <div>
      <div style={{ background: '#4a148c', color: '#fff', borderRadius: 8, padding: 20, marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22 }}>🧬 Hereditary Lysosomal Storage Disease Atlas</h2>
        <p style={{ margin: '8px 0 0', opacity: 0.9, fontSize: 13 }}>{data.subtitle}</p>
        <div style={{ marginTop: 12, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <Badge text={`${data.total_patients} Patients`} color='#1a0033' />
          <Badge text={`8 Genes`} color='#880e4f' />
          <Badge text={`Seeds ${data.seeds}`} color='#1b5e20' />
        </div>
      </div>
      <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <b>🚨 Key Clinical Rules:</b>
        <ul style={{ margin: '8px 0 0', paddingLeft: 20, fontSize: 13 }}>
          <li><b>GBA/Gaucher:</b> Splenomegaly + thrombocytopenia + bone crises TRIAD — ERT imiglucerase — SRT eliglustat (CYP2D6 mandatory) — GBA heterozygotes: 5x Parkinson risk</li>
          <li><b>SMPD1/Niemann-Pick B:</b> Olipudase alfa FDA2022 TYPE B (non-neuronopathic) ONLY — sea-blue histiocytes BM biopsy PATHOGNOMONIC — Type A: supportive only</li>
          <li><b>NPC1/Niemann-Pick C:</b> VSGP (vertical supranuclear gaze palsy) PATHOGNOMONIC — gelastic cataplexy PATHOGNOMONIC — filipin staining fibroblasts PATHOGNOMONIC — miglustat slows progression</li>
          <li><b>HEXA/Tay-Sachs:</b> Cherry-red spot fundoscopy PATHOGNOMONIC — Ashkenazi carrier 1/30 — NO ERT available — gene therapy trials</li>
          <li><b>GALC/Krabbe:</b> Psychosine elevated PATHOGNOMONIC — HSCT effective ONLY presymptomatic NBS-detected — after symptom onset: no neurological benefit</li>
          <li><b>ARSA/MLD:</b> Libmeldy EMA2020/FDA2024 presymptomatic/early-symptomatic ONLY — TIMING CRITICAL — pseudodeficiency allele: gene sequencing MANDATORY</li>
          <li><b>GLB1:</b> SAME GENE causes GM1 gangliosidosis (null mutations) AND Morquio B (missense) — C1-C2 instability Morquio B: surveillance MANDATORY</li>
          <li><b>ASAH1/Farber:</b> TRIAD: painful nodules + joint contractures + hoarseness PATHOGNOMONIC — HSCT improves somatic NOT neurological — often misdiagnosed as JIA</li>
        </ul>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 12, marginBottom: 20 }}>
        {(data.genes || []).map(gene => {
          const s = summary[gene] || {};
          const color = GENE_COLORS[gene] || '#555';
          return (
            <div key={gene} style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 12 }}>
              <div style={{ background: color, color: '#fff', padding: '4px 10px', borderRadius: 4, fontWeight: 700, marginBottom: 8 }}>{gene}</div>
              <div style={{ fontSize: 11 }}><b>Age Dx:</b> {s.mean_age_dx} yr</div>
              <div style={{ fontSize: 11 }}><b>Neuro:</b> {s.neuropathy_n}/40</div>
              <div style={{ fontSize: 11 }}><b>ERT/Tx:</b> {s.ert_treated_n}/40</div>
              <div style={{ fontSize: 11 }}><b>LSD:</b> {s.lsd_type}</div>
            </div>
          );
        })}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={{ background: '#e8f5e9', border: '2px solid #1b5e20', borderRadius: 8, padding: 16 }}>
          <b>💊 ERT Available:</b>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.ert_available || []).map(t => (
              <Badge key={t} text={t} color='#1b5e20' />
            ))}
          </div>
          <div style={{ marginTop: 10 }}><b>💊 SRT Available:</b></div>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.srt_available || []).map(t => (
              <Badge key={t} text={t} color='#0d47a1' />
            ))}
          </div>
          <div style={{ marginTop: 10 }}><b>🧬 Gene Therapy Approved:</b></div>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.gene_therapy_approved || []).map(t => (
              <Badge key={t} text={t} color='#e65100' />
            ))}
          </div>
        </div>
        <div style={{ background: '#fce4ec', border: '2px solid #880e4f', borderRadius: 8, padding: 16 }}>
          <b>👁 Pathognomonic Eye Findings:</b>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.pathognomonic_eye_findings || []).map(t => (
              <Badge key={t} text={t} color='#b71c1c' />
            ))}
          </div>
          <div style={{ marginTop: 10 }}><b>🏥 HSCT effective presymptomatic:</b></div>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.hsct_effective_presymptomatic || []).map(t => (
              <Badge key={t} text={t} color='#37474f' />
            ))}
          </div>
          <div style={{ marginTop: 10 }}><b>⚠ No disease-modifying therapy:</b></div>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.no_disease_modifying_therapy || []).map(t => (
              <Badge key={t} text={t} color='#880e4f' />
            ))}
          </div>
          <div style={{ marginTop: 10 }}><b>🔀 Same gene — two diseases:</b></div>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.same_gene_two_diseases || []).map(t => (
              <Badge key={t} text={t} color='#004d40' />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function GeneTableTab({ breakdown }) {
  if (!breakdown) return <div>Loading...</div>;
  const genes = Object.keys(breakdown.gene_data || {});
  return (
    <div>
      <h3>Per-Gene Cohort Summary (320 patients, 8 × 40)</h3>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr style={{ background: '#4a148c', color: '#fff' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gene</th>
              <th style={{ padding: '8px 12px' }}>Disease</th>
              <th style={{ padding: '8px 12px' }}>Inh</th>
              <th style={{ padding: '8px 12px' }}>Age Dx</th>
              <th style={{ padding: '8px 12px' }}>Neuro</th>
              <th style={{ padding: '8px 12px' }}>ERT/Tx</th>
              <th style={{ padding: '8px 12px' }}>HSCT</th>
              <th style={{ padding: '8px 12px', textAlign: 'left' }}>Key Therapy</th>
            </tr>
          </thead>
          <tbody>
            {genes.map((gene, idx) => {
              const d = breakdown.gene_data[gene];
              const color = GENE_COLORS[gene] || '#555';
              const inh = breakdown.inheritance_map?.[gene] || '?';
              const curative = breakdown.therapy_curative?.[gene] || '';
              const diseaseMap = {
                GBA: 'Gaucher', SMPD1: 'Niemann-Pick-A/B', NPC1: 'Niemann-Pick-C',
                HEXA: 'Tay-Sachs', GALC: 'Krabbe', ARSA: 'MLD',
                GLB1: 'GM1/Morquio-B', ASAH1: 'Farber',
              };
              return (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                  <td style={{ padding: '6px 12px' }}>
                    <span style={{ background: color, color: '#fff', padding: '2px 8px', borderRadius: 4, fontWeight: 700 }}>{gene}</span>
                  </td>
                  <td style={{ padding: '6px 12px', fontWeight: 600 }}>{diseaseMap[gene]}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', fontSize: 11 }}>{inh}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.mean_age_dx} yr</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.neuropathy_n}/40</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.ert_treated_n}/40</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.hsct_n}/40</td>
                  <td style={{ padding: '6px 12px', fontSize: 11, maxWidth: 280 }}>{curative}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ClinicalAtlasTab({ breakdown }) {
  if (!breakdown) return <div>Loading...</div>;
  return (
    <div>
      <h3>Clinical Atlas — 8 LSD Gene Profiles</h3>
      {(breakdown.atlas_genes || []).map(g => (
        <GeneCard
          key={g.gene}
          gene={g.gene}
          color={GENE_COLORS[g.gene] || '#555'}
          info={GENE_INFO[g.gene]}
          breakdown={breakdown}
        />
      ))}
    </div>
  );
}

function DefinitionsTab({ defs }) {
  if (!defs) return <div>Loading...</div>;
  return (
    <div>
      <h3>Definitions &amp; Critical Clinical Rules</h3>
      <div style={{ background: '#fce4ec', border: '2px solid #c62828', borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <b>⚠ Critical Clinical Rules</b>
        {Object.entries(defs.critical_clinical_rules || {}).map(([k, v]) => (
          <div key={k} style={{ marginTop: 10, borderTop: '1px solid #ef9a9a', paddingTop: 10 }}>
            <div style={{ fontWeight: 700, color: '#c62828', marginBottom: 4 }}>{k.replace(/_/g, '-')}</div>
            <div style={{ fontSize: 13 }}>{v}</div>
          </div>
        ))}
      </div>
      <h4>Glossary</h4>
      {Object.entries(defs.definitions || {}).map(([term, defn]) => (
        <div key={term} style={{ marginBottom: 12, borderLeft: '3px solid #4a148c', paddingLeft: 12 }}>
          <div style={{ fontWeight: 700, color: '#4a148c' }}>{term}</div>
          <div style={{ fontSize: 13 }}>{defn}</div>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryLysosomalStorageDiseaseAtlasPage() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24, fontFamily: 'sans-serif' }}>
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map((t, i) => (
          <button key={t} onClick={() => setTab(i)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: tab === i ? '#4a148c' : '#ede7f6', color: tab === i ? '#fff' : '#333',
            fontWeight: tab === i ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>
      {loading && <div>Loading atlas data…</div>}
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}
      {!loading && !error && (
        <>
          {tab === 0 && <OverviewTab data={overview} />}
          {tab === 1 && <GeneTableTab breakdown={breakdown} />}
          {tab === 2 && <ClinicalAtlasTab breakdown={breakdown} />}
          {tab === 3 && <DefinitionsTab defs={defs} />}
        </>
      )}
    </div>
  );
}
