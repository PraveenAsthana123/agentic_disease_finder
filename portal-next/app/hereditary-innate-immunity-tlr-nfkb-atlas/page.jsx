'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-innate-immunity-tlr-nfkb-atlas';
const TABS = ['Overview', 'Gene Table', 'Pathway Map', 'Definitions'];

const GENE_COLORS = {
  'IRAK4':   '#0d47a1',  // deep blue         -- AR; pyogenic bacteria; NO fever NO CRP; improves age 14
  'MYD88':   '#1b5e20',  // dark green        -- AR; identical IRAK4; somatic GOF = lymphoma driver
  'IKBKG':   '#880e4f',  // deep magenta      -- XL NEMO; EDA-ID conical teeth; BCG CI; anhydrosis
  'NFKBIA':  '#e65100',  // burnt orange      -- AD-GOF IkBa; more severe than NEMO; herpesvirus fatal
  'IRF3':    '#4a148c',  // deep purple       -- AD-LOF; HSE in healthy host; CNS-specific
  'TLR3':    '#006064',  // dark teal         -- AD/AR; most common genetic HSE; children 3-15y
  'UNC93B1': '#827717',  // dark olive        -- AR; broadest endosomal TLR; first HSE gene 2006
  'IRF7':    '#b71c1c',  // deep red          -- AR; life-threatening influenza PATHOGNOMONIC
};

const GENE_INFO = {
  'IRAK4':   { full: 'IRAK4-460aa / AR-IRAK4-Deficiency / Pyogenic-No-Fever-No-CRP-PATHOGNOMONIC / S-pneumoniae-S-aureus / Penicillin-Prophylaxis / Improves-Age-14', locus: '12q12',    size: '460 aa / 52 kDa',  inh: 'AR' },
  'MYD88':   { full: 'MYD88-296aa / AR-MyD88-Deficiency / Identical-IRAK4-Phenotype / Somatic-GOF-p-L265P-Lymphoma-Driver / Penicillin-Prophylaxis / Improves-Age-14', locus: '3p22.2',  size: '296 aa / 33 kDa',  inh: 'AR' },
  'IKBKG':   { full: 'IKBKG-NEMO-419aa / XL-EDA-ID / Anhydrosis-Conical-Teeth-PATHOGNOMONIC / Mycobacteria-Candida / BCG-ABSOLUTELY-CI / Males-Only', locus: 'Xq28',    size: '419 aa / 48 kDa',  inh: 'XL' },
  'NFKBIA':  { full: 'NFKBIA-IkBa-317aa / AD-GOF / EDA-ID-MORE-SEVERE-Than-NEMO / EBV-CMV-VZV-Fatal / Mycobacteria / BCG-ABSOLUTELY-CI / Both-Sexes', locus: '14q13.2', size: '317 aa / 36 kDa',  inh: 'AD-GOF' },
  'IRF3':    { full: 'IRF3-427aa / AD-LOF / Herpes-Simplex-Encephalitis-Only / CNS-Selective-IFN-Beta-Defect / Systemic-Immunity-Intact / Aciclovir-Prophylaxis', locus: '19q13.33', size: '427 aa / 47 kDa', inh: 'AD-LOF' },
  'TLR3':    { full: 'TLR3-904aa / AD-LOF-or-AR / MOST-COMMON-Genetic-HSE / Children-3-15y / Poly-IC-Fibroblast-Assay-Diagnostic / Aciclovir-Prophylaxis', locus: '4q35.1',   size: '904 aa / 104 kDa', inh: 'AD/AR' },
  'UNC93B1': { full: 'UNC93B1-975aa / AR / Broadest-TLR3-TLR7-TLR8-TLR9-All-Absent / HSE-Primary-Phenotype / First-Described-Genetic-HSE-2006 / pDC-IFNa-Impaired', locus: '11q13.3', size: '975 aa / 110 kDa', inh: 'AR' },
  'IRF7':    { full: 'IRF7-457aa / AR / Life-Threatening-Influenza-PATHOGNOMONIC / IFN-Alpha-Amplification-Absent / Annual-Flu-Vaccine-MANDATORY / Oseltamivir-Prophylaxis', locus: '11p15.5', size: '457 aa / 52 kDa',  inh: 'AR' },
};

const PHENOTYPE_ROWS = [
  { gene: 'IRAK4',   axis: 'NF-κB', pyogenic: '✓✓✓', no_inflam: '✓✓✓', eda: '−',    myco: '±',   hse: '−',   flu: '−',   age_imp: '✓✓✓', unique: 'No fever/CRP/ESR with invasive bacteraemia — PATHOGNOMONIC; improves markedly by age 14' },
  { gene: 'MYD88',   axis: 'NF-κB', pyogenic: '✓✓✓', no_inflam: '✓✓✓', eda: '−',    myco: '±',   hse: '−',   flu: '−',   age_imp: '✓✓✓', unique: 'Identical to IRAK4 clinically; somatic GOF same gene = DLBCL/WM driver (opposite biology)' },
  { gene: 'IKBKG',   axis: 'NF-κB', pyogenic: '✓✓',  no_inflam: '✓',   eda: '✓✓✓', myco: '✓✓✓', hse: '±',   flu: '−',   age_imp: '±',   unique: 'NEMO XL males; HED triad PATHOGNOMONIC; BCG ABSOLUTELY CI; anhydrosis → hyperthermia fatal' },
  { gene: 'NFKBIA',  axis: 'NF-κB', pyogenic: '✓✓',  no_inflam: '✓',   eda: '✓✓✓', myco: '✓✓✓', hse: '✓✓', flu: '±',   age_imp: '±',   unique: 'AD-GOF IκBα; MORE SEVERE than NEMO; EBV/CMV/VZV fatal; HSCT more commonly needed' },
  { gene: 'IRF3',    axis: 'IFN',   pyogenic: '−',   no_inflam: '−',   eda: '−',    myco: '−',   hse: '✓✓✓',flu: '−',   age_imp: '−',   unique: 'HSE in otherwise healthy host; CNS-selective; systemic immunity intact; aciclovir lifelong' },
  { gene: 'TLR3',    axis: 'IFN',   pyogenic: '−',   no_inflam: '−',   eda: '−',    myco: '−',   hse: '✓✓✓',flu: '−',   age_imp: '−',   unique: 'MOST COMMON genetic HSE cause; children 3–15y; poly-I:C fibroblast assay diagnostic' },
  { gene: 'UNC93B1', axis: 'IFN',   pyogenic: '±',   no_inflam: '−',   eda: '−',    myco: '−',   hse: '✓✓✓',flu: '±',   age_imp: '−',   unique: 'Broadest endosomal TLR defect (TLR3+7+8+9); first described HSE gene 2006; pDC IFN-α impaired' },
  { gene: 'IRF7',    axis: 'IFN',   pyogenic: '−',   no_inflam: '−',   eda: '−',    myco: '−',   hse: '−',   flu: '✓✓✓',age_imp: '−',   unique: 'LIFE-THREATENING INFLUENZA PATHOGNOMONIC; IFN-α amplification absent; flu vaccine MANDATORY' },
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
  const axisLabel = ['IRAK4','MYD88','IKBKG','NFKBIA'].includes(gene) ? 'NF-κB Axis' : 'Type-I IFN Axis';
  const axisColor = ['IRAK4','MYD88','IKBKG','NFKBIA'].includes(gene) ? '#0d47a1' : '#6a1b9a';
  return (
    <div style={{ border: `2px solid ${color}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: color, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div style={{ background: axisColor, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>{axisLabel}</div>
        <div>
          <div style={{ fontWeight: 600 }}>{cohort?.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>{info?.full?.replace(/-/g, ' ')}</div>
      {cohort && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 8 }}>
          <Badge text={`n=${cohort.n}`} color={color} />
          <Badge text={`Onset ${cohort.mean_onset_months?.toFixed(0)}mo`} color='#555' />
          <Badge text={`Dx lag ${cohort.mean_dx_lag_yrs?.toFixed(1)}y`} color='#888' />
          {cohort.invasive_bacterial_n > 0 && <Badge text={`Invasive-bact ${cohort.invasive_bacterial_n}/${cohort.n}`} color='#1565c0' />}
          {cohort.no_fever_response_n > 0 && <Badge text={`No-fever ${cohort.no_fever_response_n}/${cohort.n}`} color='#b71c1c' />}
          {cohort.eda_triad_n > 0 && <Badge text={`EDA-triad ${cohort.eda_triad_n}/${cohort.n}`} color='#880e4f' />}
          {cohort.mycobacterial_n > 0 && <Badge text={`Myco ${cohort.mycobacterial_n}/${cohort.n}`} color='#4e342e' />}
          {cohort.hse_n > 0 && <Badge text={`HSE ${cohort.hse_n}/${cohort.n}`} color='#4a148c' />}
          {cohort.influenza_severe_n > 0 && <Badge text={`Flu-severe ${cohort.influenza_severe_n}/${cohort.n}`} color='#c62828' />}
          {cohort.age_improvement_n > 0 && <Badge text={`Age-improve ${cohort.age_improvement_n}/${cohort.n}`} color='#2e7d32' />}
        </div>
      )}
      {cohort?.pathognomonic && (
        <div style={{ background: `${color}15`, borderLeft: `3px solid ${color}`, padding: '6px 10px', fontSize: 12, borderRadius: '0 4px 4px 0' }}>
          <strong>Pathognomonic / Distinguisher:</strong> {cohort.pathognomonic}
        </div>
      )}
    </div>
  );
}

export default function HeredInnateImmunityTlrNfkbAtlas() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    const ep = tab === 'Definitions' ? 'definitions' : tab === 'Gene Table' ? 'breakdown' : 'overview';
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => {
        if (tab === 'Overview' || tab === 'Pathway Map') setOverview(d);
        else if (tab === 'Gene Table') setBreakdown(d);
        else setDefinitions(d);
        setLoading(false);
      })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [tab]);

  const cohortMap = Object.fromEntries(
    (breakdown?.cohorts || overview?.cohort_summary || []).map(c => [c.gene, c])
  );

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1400, margin: '0 auto', padding: 24 }}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4 }}>
          🧬 Hereditary Innate Immunity TLR–NF-κB Atlas
        </h1>
        <div style={{ color: '#555', fontSize: 13 }}>
          Complete 8-Gene Reference · IRAK4 / MYD88 / NEMO(IKBKG) / IκBα(NFKBIA) / IRF3 / TLR3 / UNC93B1 / IRF7 ·
          NF-κB Axis + Type-I IFN Axis · 320 patients (n=40/gene) · Seeds 3886–3893
        </div>
        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          <Badge text="NF-κB Axis: IRAK4/MYD88/NEMO/IκBα" color="#0d47a1" />
          <Badge text="IFN Axis: IRF3/TLR3/UNC93B1/IRF7" color="#6a1b9a" />
          <Badge text="No-fever pyogenic bacteraemia PATHOGNOMONIC" color="#1565c0" />
          <Badge text="HSE in healthy child PATHOGNOMONIC" color="#4a148c" />
          <Badge text="Fatal influenza PATHOGNOMONIC (IRF7)" color="#b71c1c" />
          <Badge text="EDA-ID: conical teeth PATHOGNOMONIC" color="#880e4f" />
          <Badge text="BCG ABSOLUTELY CI (NEMO + IκBα)" color="#c62828" />
          <Badge text="Improves age 14 (IRAK4/MYD88 only)" color="#2e7d32" />
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, borderBottom: '2px solid #e0e0e0', paddingBottom: 8 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: '6px 6px 0 0', border: 'none', cursor: 'pointer',
            background: tab === t ? '#0d47a1' : '#f5f5f5',
            color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400, fontSize: 14,
          }}>{t}</button>
        ))}
      </div>

      {loading && <div style={{ color: '#888', padding: 40, textAlign: 'center' }}>Loading…</div>}
      {error && <div style={{ color: '#c62828', padding: 16, background: '#ffebee', borderRadius: 4 }}>Error: {error}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && !loading && (
        <div>
          <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 8px', color: '#0d47a1' }}>{overview.title}</h3>
            <div style={{ color: '#555', marginBottom: 12 }}>{overview.subtitle}</div>
            <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
              <div><strong>{overview.total_patients}</strong> total patients</div>
              <div><strong>{overview.cohort_per_gene}</strong> per gene</div>
              <div><strong>Seeds:</strong> {overview.seeds}</div>
            </div>
          </div>

          {/* Two pathway groups */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <div style={{ background: '#e8eaf6', borderRadius: 8, padding: 16 }}>
              <h4 style={{ color: '#0d47a1', margin: '0 0 8px' }}>🏥 NF-κB Axis Defects</h4>
              <div style={{ fontSize: 13, marginBottom: 8 }}>{overview.nfkb_axis_phenotype}</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {['IRAK4','MYD88','IKBKG','NFKBIA'].map(g => (
                  <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', padding: '2px 8px', borderRadius: 4, fontSize: 12 }}>{g}</span>
                ))}
              </div>
            </div>
            <div style={{ background: '#f3e5f5', borderRadius: 8, padding: 16 }}>
              <h4 style={{ color: '#6a1b9a', margin: '0 0 8px' }}>🦠 Type-I IFN Axis Defects</h4>
              <div style={{ fontSize: 13, marginBottom: 8 }}>{overview.ifn_axis_phenotype}</div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                {['IRF3','TLR3','UNC93B1','IRF7'].map(g => (
                  <span key={g} style={{ background: GENE_COLORS[g], color: '#fff', padding: '2px 8px', borderRadius: 4, fontSize: 12 }}>{g}</span>
                ))}
              </div>
            </div>
          </div>

          <div style={{ background: '#fff8e1', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h4 style={{ margin: '0 0 8px', color: '#e65100' }}>Key Clinical Clues</h4>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
              {(overview.key_clinical_clues || []).map((c, i) => <li key={i} style={{ marginBottom: 4 }}>{c}</li>)}
            </ul>
          </div>

          <div style={{ background: '#f3e5f5', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <h4 style={{ margin: '0 0 8px', color: '#6a1b9a' }}>HLH-2004 Diagnostic Workup</h4>
            <ul style={{ margin: 0, paddingLeft: 20, fontSize: 13 }}>
              {(overview.diagnostic_workup || []).map((w, i) => <li key={i} style={{ marginBottom: 4 }}>{w}</li>)}
            </ul>
          </div>

          <h3 style={{ marginBottom: 12 }}>Gene Cards</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {GENES.map(g => (
              <GeneCard key={g} gene={g} color={GENE_COLORS[g]} info={GENE_INFO[g]}
                cohort={(overview.cohort_summary || []).find(c => c.gene === g)} />
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && !loading && (
        <div>
          <h3 style={{ marginBottom: 16 }}>Gene × Phenotype Matrix</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#1a237e', color: '#fff' }}>
                  <th style={{ padding: '10px 12px', textAlign: 'left', minWidth: 80 }}>Gene</th>
                  <th style={{ padding: '10px 8px', textAlign: 'center' }}>Axis</th>
                  <th style={{ padding: '10px 8px', textAlign: 'center' }}>Pyogenic</th>
                  <th style={{ padding: '10px 8px', textAlign: 'center' }}>No-Inflam</th>
                  <th style={{ padding: '10px 8px', textAlign: 'center' }}>EDA-Triad</th>
                  <th style={{ padding: '10px 8px', textAlign: 'center' }}>Myco</th>
                  <th style={{ padding: '10px 8px', textAlign: 'center' }}>HSE</th>
                  <th style={{ padding: '10px 8px', textAlign: 'center' }}>Flu-Severe</th>
                  <th style={{ padding: '10px 8px', textAlign: 'center' }}>Age-Improve</th>
                  <th style={{ padding: '10px 12px', textAlign: 'left', minWidth: 300 }}>Unique Feature</th>
                </tr>
              </thead>
              <tbody>
                {PHENOTYPE_ROWS.map((row, i) => (
                  <tr key={row.gene} style={{ background: i % 2 ? '#f5f5f5' : '#fff' }}>
                    <td style={{ padding: '10px 12px', fontWeight: 700, color: GENE_COLORS[row.gene] }}>{row.gene}</td>
                    <td style={{ padding: '10px 8px', textAlign: 'center' }}>
                      <span style={{ background: row.axis === 'NF-κB' ? '#0d47a1' : '#6a1b9a', color: '#fff', padding: '1px 6px', borderRadius: 3, fontSize: 11 }}>{row.axis}</span>
                    </td>
                    <td style={{ padding: '10px 8px', textAlign: 'center' }}>{row.pyogenic}</td>
                    <td style={{ padding: '10px 8px', textAlign: 'center' }}>{row.no_inflam}</td>
                    <td style={{ padding: '10px 8px', textAlign: 'center' }}>{row.eda}</td>
                    <td style={{ padding: '10px 8px', textAlign: 'center' }}>{row.myco}</td>
                    <td style={{ padding: '10px 8px', textAlign: 'center' }}>{row.hse}</td>
                    <td style={{ padding: '10px 8px', textAlign: 'center' }}>{row.flu}</td>
                    <td style={{ padding: '10px 8px', textAlign: 'center' }}>{row.age_imp}</td>
                    <td style={{ padding: '10px 12px', fontSize: 12, color: '#444' }}>{row.unique}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3 style={{ marginTop: 32, marginBottom: 16 }}>Per-Gene Cohort Cards</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {GENES.map(g => (
              <GeneCard key={g} gene={g} color={GENE_COLORS[g]} info={GENE_INFO[g]} cohort={cohortMap[g]} />
            ))}
          </div>
        </div>
      )}

      {/* PATHWAY MAP TAB */}
      {tab === 'Pathway Map' && overview && !loading && (
        <div>
          <h3 style={{ marginBottom: 16 }}>Innate Immunity Signalling Pathway — Disease Gene Map</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            {/* NF-kB arm */}
            <div style={{ background: '#e8eaf6', borderRadius: 12, padding: 20 }}>
              <h4 style={{ color: '#0d47a1', margin: '0 0 16px' }}>NF-κB Signalling Arm</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 13 }}>
                <div style={{ background: '#fff', padding: 10, borderRadius: 6, borderLeft: '4px solid #42a5f5' }}>
                  <strong>TLR1/2/4/5/6/7/8/9 + IL-1R/IL-18R/IL-33R</strong><br/>
                  Pattern recognition → TIR domain activation
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓</div>
                <div style={{ background: GENE_COLORS['MYD88'] + '22', padding: 10, borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS['MYD88']}` }}>
                  <strong style={{ color: GENE_COLORS['MYD88'] }}>MYD88</strong> — TIR-domain adaptor<br/>
                  <span style={{ fontSize: 11, color: '#666' }}>AR LOF → no Myddosome assembly</span>
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓</div>
                <div style={{ background: GENE_COLORS['IRAK4'] + '22', padding: 10, borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS['IRAK4']}` }}>
                  <strong style={{ color: GENE_COLORS['IRAK4'] }}>IRAK4</strong> — Myddosome kinase<br/>
                  <span style={{ fontSize: 11, color: '#666' }}>AR LOF → IRAK1/2 not phosphorylated</span>
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓ TRAF6 → IKK complex ↓</div>
                <div style={{ background: GENE_COLORS['IKBKG'] + '22', padding: 10, borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS['IKBKG']}` }}>
                  <strong style={{ color: GENE_COLORS['IKBKG'] }}>IKBKG/NEMO</strong> — IKK scaffold<br/>
                  <span style={{ fontSize: 11, color: '#666' }}>XL LOF → IKKα/β inactive + EDA triad</span>
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓ IKKβ phosphorylates IκBα ↓</div>
                <div style={{ background: GENE_COLORS['NFKBIA'] + '22', padding: 10, borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS['NFKBIA']}` }}>
                  <strong style={{ color: GENE_COLORS['NFKBIA'] }}>NFKBIA/IκBα</strong> — NF-κB inhibitor<br/>
                  <span style={{ fontSize: 11, color: '#666' }}>AD GOF → resistant to phosphorylation → NF-κB locked in cytoplasm</span>
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓</div>
                <div style={{ background: '#fff', padding: 10, borderRadius: 6, borderLeft: '4px solid #42a5f5' }}>
                  <strong>NF-κB → Nucleus</strong><br/>
                  TNF, IL-6, IL-8, IL-1β, CXCL1 → inflammatory response
                </div>
              </div>
            </div>

            {/* Type I IFN arm */}
            <div style={{ background: '#f3e5f5', borderRadius: 12, padding: 20 }}>
              <h4 style={{ color: '#6a1b9a', margin: '0 0 16px' }}>Type-I IFN Signalling Arm</h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 13 }}>
                <div style={{ background: '#fff', padding: 10, borderRadius: 6, borderLeft: '4px solid #ce93d8' }}>
                  <strong>HSV-1 dsRNA in CNS endosomes</strong><br/>
                  Neurons, astrocytes detect viral replication intermediates
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓</div>
                <div style={{ background: GENE_COLORS['UNC93B1'] + '22', padding: 10, borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS['UNC93B1']}` }}>
                  <strong style={{ color: GENE_COLORS['UNC93B1'] }}>UNC93B1</strong> — TLR chaperone<br/>
                  <span style={{ fontSize: 11, color: '#666' }}>AR LOF → TLR3/7/8/9 all stranded in ER</span>
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓</div>
                <div style={{ background: GENE_COLORS['TLR3'] + '22', padding: 10, borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS['TLR3']}` }}>
                  <strong style={{ color: GENE_COLORS['TLR3'] }}>TLR3</strong> — dsRNA sensor (endosome)<br/>
                  <span style={{ fontSize: 11, color: '#666' }}>AD/AR LOF → no dsRNA detection → TRIF not activated</span>
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓ TRIF → TBK1/IKKε ↓</div>
                <div style={{ background: GENE_COLORS['IRF3'] + '22', padding: 10, borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS['IRF3']}` }}>
                  <strong style={{ color: GENE_COLORS['IRF3'] }}>IRF3</strong> — immediate IFN-β driver<br/>
                  <span style={{ fontSize: 11, color: '#666' }}>AD LOF → no IFN-β transcription in CNS cells → HSE</span>
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓ IFN-β → IFNAR → STAT1/2/IRF9 ↓</div>
                <div style={{ background: GENE_COLORS['IRF7'] + '22', padding: 10, borderRadius: 6, borderLeft: `4px solid ${GENE_COLORS['IRF7']}` }}>
                  <strong style={{ color: GENE_COLORS['IRF7'] }}>IRF7</strong> — IFN-α amplification master<br/>
                  <span style={{ fontSize: 11, color: '#666' }}>AR LOF → IFN-α surge fails → influenza uncontrolled</span>
                </div>
                <div style={{ textAlign: 'center', color: '#888' }}>↓</div>
                <div style={{ background: '#fff', padding: 10, borderRadius: 6, borderLeft: '4px solid #ce93d8' }}>
                  <strong>IFN-α (subtypes 1–14) + IFN-β → antiviral state</strong><br/>
                  ISGs, PKR, OAS/RNaseL, MX1 — viral clearance
                </div>
              </div>
            </div>
          </div>

          <div style={{ background: '#fff3e0', borderRadius: 8, padding: 16, marginTop: 20 }}>
            <h4 style={{ margin: '0 0 8px', color: '#e65100' }}>Age-Improvement Hallmark (IRAK4 + MYD88 only)</h4>
            <p style={{ margin: 0, fontSize: 13 }}>
              {overview.age_improvement_hallmark}
            </p>
          </div>
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && definitions && !loading && (
        <div>
          <h3 style={{ marginBottom: 16 }}>Gene Definitions & Clinical Pearls</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {GENES.map(g => (
              <div key={g} style={{ border: `2px solid ${GENE_COLORS[g]}`, borderRadius: 8, padding: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
                  <div style={{ background: GENE_COLORS[g], color: '#fff', borderRadius: 6, padding: '4px 12px', fontWeight: 700, fontSize: 16 }}>{g}</div>
                  <div style={{ fontSize: 12, color: '#666' }}>
                    {definitions.gene_metadata?.[g]?.locus} · {definitions.gene_metadata?.[g]?.aa}aa · {definitions.gene_metadata?.[g]?.inh}
                  </div>
                </div>
                <div style={{ fontSize: 13, color: '#333', lineHeight: 1.6 }}>
                  {definitions.definitions?.[g]}
                </div>
              </div>
            ))}
          </div>

          <div style={{ marginTop: 24 }}>
            <h3 style={{ marginBottom: 12 }}>Key Distinguishers</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              {Object.entries(definitions.key_distinguishers || {}).map(([k, v]) => (
                <div key={k} style={{ background: '#f8f9fa', borderRadius: 6, padding: 12, borderLeft: '3px solid #0d47a1' }}>
                  <div style={{ fontWeight: 600, fontSize: 12, color: '#0d47a1', marginBottom: 4 }}>{k.replace(/_/g, ' ')}</div>
                  <div style={{ fontSize: 12, color: '#444' }}>{v}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 24 }}>
            <h3 style={{ marginBottom: 12 }}>Diagnostic Pathway</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {(definitions.diagnostic_pathway || []).map((step, i) => (
                <div key={i} style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                  <div style={{ background: '#0d47a1', color: '#fff', borderRadius: '50%', width: 24, height: 24, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 700, flexShrink: 0 }}>{i + 1}</div>
                  <div style={{ fontSize: 13, color: '#333', paddingTop: 3 }}>{step}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

const GENES = ["IRAK4", "MYD88", "IKBKG", "NFKBIA", "IRF3", "TLR3", "UNC93B1", "IRF7"];
