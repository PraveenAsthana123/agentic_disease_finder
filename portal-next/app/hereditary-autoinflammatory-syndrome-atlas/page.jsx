'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-autoinflammatory-syndrome-atlas';
const TABS = ['Overview', 'Gene Table', 'Syndrome Classification', 'Definitions'];

const GENE_COLORS = {
  'MEFV':     '#c62828',  // deep red       -- FMF most common; colchicine CURATIVE
  'NLRP3':    '#e65100',  // burnt orange   -- CAPS; IL-1β sole mediator; anakinra HIGHLY EFFECTIVE
  'TNFRSF1A': '#6a1b9a',  // deep purple    -- TRAPS; attacks >5d; migratory rash PATHOGNOMONIC
  'MVK':      '#1565c0',  // deep blue      -- MKD/HIDS; IgD elevated 80%; vaccination trigger
  'NOD2':     '#1b5e20',  // dark green     -- Blau syndrome; granulomatous triad PATHOGNOMONIC
  'PSTPIP1':  '#4a148c',  // indigo         -- PAPA; pyoderma + sterile arthritis + acne PATHOGNOMONIC
  'ADA2':     '#b71c1c',  // crimson red    -- DADA2; childhood strokes; TNF-i PREVENTS strokes
  'NLRP12':   '#37474f',  // blue-grey      -- FCAS2; cold-triggered; milder CAPS
};

const GENE_INFO = {
  'MEFV':     { full: 'Pyrin-441aa / FMF-most-common-periodic-fever / AR-biallelic / colchicine-CURATIVE-90pct / attacks-12-72h / AA-amyloidosis-untreated', locus: '16p13.3', size: '441 aa / 53 kDa', inh: 'AR' },
  'NLRP3':    { full: 'Cryopyrin-1036aa / CAPS-spectrum-FCAS-MWS-NOMID / AD-GOF / IL1β-sole-mediator / SNHL-PATHOGNOMONIC-MWS / anakinra-canakinumab-HIGHLY-EFFECTIVE', locus: '1q44', size: '1036 aa / 118 kDa', inh: 'AD GOF' },
  'TNFRSF1A': { full: 'TNF-receptor-1-455aa / TRAPS / AD / attacks-gt5d-KEY-distinction / migratory-rash-PATHOGNOMONIC / periorbital-oedema-raccoon-eyes / IL1i-superior', locus: '12p13.31', size: '455 aa / 51 kDa', inh: 'AD' },
  'MVK':      { full: 'Mevalonate-kinase-396aa / MKD-HIDS / AR / IgD-gt100-80pct-biomarker / attacks-q4-6wks / vaccination-trigger / canakinumab-FDA2016', locus: '12q24.11', size: '396 aa / 41 kDa', inh: 'AR' },
  'NOD2':     { full: 'NOD2-1040aa / Blau-syndrome-early-onset-sarcoidosis / AD-GOF / granulomatous-triad-arthritis-uveitis-rash-PATHOGNOMONIC / onset-lt4yr / camptodactyly', locus: '16q21', size: '1040 aa / 114 kDa', inh: 'AD GOF' },
  'PSTPIP1':  { full: 'PSTPIP1-415aa / PAPA-syndrome / AD-GOF / sterile-arthritis-NOT-infection / pyoderma-gangrenosum-PATHOGNOMONIC / acne-severe / TNFi-first-line', locus: '15q24.3', size: '415 aa / 47 kDa', inh: 'AD GOF' },
  'ADA2':     { full: 'ADA2-CECR1-511aa / DADA2 / AR / childhood-strokes-without-cardiac-cause-RED-FLAG / PAN-like-vasculitis / TNFi-PREVENTS-strokes-IMMEDIATELY / HSCT-curative', locus: '22q11.1', size: '511 aa / 59 kDa', inh: 'AR' },
  'NLRP12':   { full: 'NLRP12-Monarch1-1062aa / FCAS2 / AD-LOF / cold-triggered / milder-than-NLRP3-CAPS / hearing-loss-less-consistent / anakinra-responsive', locus: '19q13.42', size: '1062 aa / 120 kDa', inh: 'AD LOF' },
};

const SYNDROME_ROWS = [
  { gene: 'MEFV',     syndrome: 'FMF',        attack_dur: '12–72 h', trigger: 'Spontaneous', key_lab: 'SAA/CRP elevated', treatment: 'Colchicine CURATIVE', amyloid: 'Yes (untreated)' },
  { gene: 'NLRP3',    syndrome: 'CAPS',       attack_dur: 'Continuous', trigger: 'Cold (FCAS)', key_lab: 'IL-18, SAA', treatment: 'Anakinra / Canakinumab', amyloid: 'Yes (MWS/NOMID)' },
  { gene: 'TNFRSF1A', syndrome: 'TRAPS',      attack_dur: '>5 days', trigger: 'Spontaneous', key_lab: 'Soluble TNFR1 low', treatment: 'Anakinra / IL-1i', amyloid: 'Yes' },
  { gene: 'MVK',      syndrome: 'MKD/HIDS',   attack_dur: '3–7 days', trigger: 'Vaccination / illness', key_lab: 'IgD >100 IU/mL (80%)', treatment: 'Canakinumab (FDA 2016)', amyloid: 'Rare' },
  { gene: 'NOD2',     syndrome: 'Blau',       attack_dur: 'Chronic', trigger: 'Continuous', key_lab: 'Granulomas on biopsy', treatment: 'MTX + Adalimumab', amyloid: 'Rare' },
  { gene: 'PSTPIP1',  syndrome: 'PAPA',       attack_dur: 'Days–weeks', trigger: 'Trauma / spontaneous', key_lab: 'Negative joint cultures', treatment: 'Infliximab / IL-1i', amyloid: 'Rare' },
  { gene: 'ADA2',     syndrome: 'DADA2',      attack_dur: 'Variable', trigger: 'Spontaneous', key_lab: 'ADA2 enzyme low', treatment: 'Etanercept IMMEDIATELY', amyloid: 'Rare' },
  { gene: 'NLRP12',   syndrome: 'FCAS2',      attack_dur: '6–24 h', trigger: 'Cold', key_lab: 'SAA/CRP elevated', treatment: 'Anakinra / Canakinumab', amyloid: 'Rare' },
];

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
  return (
    <div style={{ border: `2px solid ${bg}`, borderRadius: 8, padding: 16, marginBottom: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <div style={{ background: bg, color: '#fff', borderRadius: 6, padding: '6px 14px', fontWeight: 700, fontSize: 18 }}>{gene}</div>
        <div>
          <div style={{ fontWeight: 600 }}>{cohort.syndrome || ''}</div>
          <div style={{ fontSize: 12, color: '#666' }}>{info?.locus} · {info?.size} · {info?.inh}</div>
        </div>
      </div>
      <div style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>{info?.full?.replace(/-/g, ' ')}</div>
      {cohort.pathognomonic && (
        <div style={{ background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 4, padding: '6px 10px', fontSize: 12, marginBottom: 8 }}>
          <strong>PATHOGNOMONIC:</strong> {cohort.pathognomonic}
        </div>
      )}
      {cohort.first_line_treatment && (
        <div style={{ background: '#e8f5e9', border: '1px solid #4caf50', borderRadius: 4, padding: '6px 10px', fontSize: 12, marginBottom: 8 }}>
          <strong>Treatment:</strong> {cohort.first_line_treatment}
        </div>
      )}
      {cohort.n && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 8 }}>
          <Badge text={`n=${cohort.n}`} color={bg} />
          <Badge text={`onset ${cohort.mean_onset_age}yr`} color="#555" />
          <Badge text={`dx-lag ${cohort.mean_dx_lag_yrs}yr`} color="#888" />
          {cohort.attack_frequency_per_yr > 0 && <Badge text={`attacks ${cohort.attack_frequency_per_yr}/yr`} color="#1565c0" />}
          {cohort.arthritis_n > 0 && <Badge text={`arthritis ${cohort.arthritis_n}/40`} color="#6a1b9a" />}
          {cohort.skin_rash_n > 0 && <Badge text={`rash ${cohort.skin_rash_n}/40`} color="#e65100" />}
          {cohort.sensorineural_hearing_loss_n > 0 && <Badge text={`SNHL ${cohort.sensorineural_hearing_loss_n}/40`} color="#880e4f" />}
          {cohort.uveitis_n > 0 && <Badge text={`uveitis ${cohort.uveitis_n}/40`} color="#1b5e20" />}
          {cohort.stroke_n > 0 && <Badge text={`stroke ${cohort.stroke_n}/40`} color="#c62828" />}
          {cohort.pyoderma_gangrenosum_n > 0 && <Badge text={`PG ${cohort.pyoderma_gangrenosum_n}/40`} color="#4a148c" />}
          {cohort.igd_elevated_n > 0 && <Badge text={`IgD-high ${cohort.igd_elevated_n}/40`} color="#1565c0" />}
          {cohort.il1_blockade_n > 0 && <Badge text={`IL-1i ${cohort.il1_blockade_n}/40`} color="#b71c1c" />}
          {cohort.colchicine_response_n > 0 && <Badge text={`colchicine-resp ${cohort.colchicine_response_n}/40`} color="#2e7d32" />}
        </div>
      )}
    </div>
  );
}

export default function HAutoinflammatoryAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, df]) => { setOverview(ov); setBreakdown(br); setDefs(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div style={{ padding: 40, color: '#666' }}>Loading Hereditary Autoinflammatory Syndrome Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const genes = overview?.genes || [];

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#c62828,#6a1b9a)', color: '#fff', borderRadius: 10, padding: 24, marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>🔥 Hereditary Autoinflammatory Syndrome Atlas</h1>
        <p style={{ margin: '8px 0 0', opacity: 0.9, fontSize: 13 }}>
          Complete 8-Gene Reference · MEFV/FMF · NLRP3/CAPS · TNFRSF1A/TRAPS · MVK/MKD-HIDS · NOD2/Blau · PSTPIP1/PAPA · ADA2/DADA2 · NLRP12/FCAS2
        </p>
        <p style={{ margin: '4px 0 0', opacity: 0.75, fontSize: 12 }}>
          320 patients · 8×40 cohorts · seeds {overview?.seeds}
        </p>
      </div>

      {/* Critical Rules Banner */}
      <div style={{ background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 8, padding: 12, marginBottom: 20 }}>
        <strong>🚨 Critical Clinical Rules:</strong>
        <ul style={{ margin: '8px 0 0', paddingLeft: 20, fontSize: 12 }}>
          <li><strong>MEFV/FMF:</strong> Colchicine 0.5–2 mg/day LIFELONG = CURATIVE (90%); prevents AA amyloidosis — NEVER stop</li>
          <li><strong>NLRP3/CAPS:</strong> IL-1 blockade MANDATORY (anakinra/canakinumab) — sole effective therapy; prevents hearing loss + amyloidosis</li>
          <li><strong>TNFRSF1A/TRAPS:</strong> Attacks &gt;5 days = KEY distinction from FMF (12–72h); migratory muscle-overlying rash PATHOGNOMONIC</li>
          <li><strong>ADA2/DADA2:</strong> Start etanercept IMMEDIATELY — prevents childhood ischaemic strokes; do NOT wait for next event</li>
          <li><strong>NOD2/Blau:</strong> Granulomatous triad (arthritis + uveitis + rash) onset &lt;4yr PATHOGNOMONIC; ophthalmology mandatory (blindness risk)</li>
          <li><strong>PSTPIP1/PAPA:</strong> Sterile arthritis (negative cultures) + pyoderma gangrenosum + acne = PAPA; do NOT debride PG (pathergy worsens)</li>
        </ul>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{ padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: activeTab === t ? '#c62828' : '#eee', color: activeTab === t ? '#fff' : '#333',
              fontWeight: activeTab === t ? 700 : 400 }}>
            {t}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'Overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, marginBottom: 20 }}>
            <div style={{ background: '#c62828', color: '#fff', borderRadius: 8, padding: 16, textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 700 }}>{overview?.total_patients}</div>
              <div style={{ fontSize: 12 }}>Total Patients</div>
            </div>
            <div style={{ background: '#1565c0', color: '#fff', borderRadius: 8, padding: 16, textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 700 }}>{genes.length}</div>
              <div style={{ fontSize: 12 }}>Genes / Syndromes</div>
            </div>
            <div style={{ background: '#2e7d32', color: '#fff', borderRadius: 8, padding: 16, textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 700 }}>40</div>
              <div style={{ fontSize: 12 }}>Patients per Gene</div>
            </div>
            <div style={{ background: '#6a1b9a', color: '#fff', borderRadius: 8, padding: 16, textAlign: 'center' }}>
              <div style={{ fontSize: 28, fontWeight: 700 }}>{overview?.seeds}</div>
              <div style={{ fontSize: 12 }}>Seeds</div>
            </div>
          </div>

          {overview?.key_facts && (
            <div style={{ background: '#f5f5f5', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 12px' }}>Key Clinical Facts</h3>
              {Object.entries(overview.key_facts).map(([k, v]) => (
                <div key={k} style={{ marginBottom: 6, fontSize: 13 }}>
                  <strong>{k.replace(/_/g, ' ')}:</strong> {v}
                </div>
              ))}
            </div>
          )}

          {overview?.diagnostic_approach && (
            <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16, marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 12px' }}>Diagnostic Approach by Presentation</h3>
              {Object.entries(overview.diagnostic_approach).map(([scenario, approach]) => (
                <div key={scenario} style={{ marginBottom: 8, fontSize: 13 }}>
                  <strong style={{ color: '#1565c0' }}>{scenario.replace(/_/g, ' ')}:</strong> {approach}
                </div>
              ))}
            </div>
          )}

          <h3>Gene Overview Cards</h3>
          {genes.map(gene => (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene] || '#555'}
              info={GENE_INFO[gene]} breakdown={breakdown} />
          ))}
        </div>
      )}

      {/* Gene Table Tab */}
      {activeTab === 'Gene Table' && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
            <thead>
              <tr style={{ background: '#c62828', color: '#fff' }}>
                {['Gene', 'Syndrome', 'Locus', 'Inh', 'Onset (yr)', 'Attack Freq/yr', 'Attack Dur (h)', 'IL-1i n/40', 'Colchicine n/40', 'Arthritis n/40', 'Rash n/40', 'Pathognomonic'].map(h => (
                  <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {genes.map((gene, idx) => {
                const d = breakdown?.gene_data?.[gene] || {};
                return (
                  <tr key={gene} style={{ background: idx % 2 === 0 ? '#fff' : '#fafafa', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[gene] || '#333' }}>{gene}</td>
                    <td style={{ padding: '7px 10px' }}>{d.syndrome || '—'}</td>
                    <td style={{ padding: '7px 10px' }}>{d.locus || GENE_INFO[gene]?.locus || '—'}</td>
                    <td style={{ padding: '7px 10px' }}>{d.inheritance || '—'}</td>
                    <td style={{ padding: '7px 10px' }}>{d.mean_onset_age ?? '—'}</td>
                    <td style={{ padding: '7px 10px' }}>{d.attack_frequency_per_yr ?? '—'}</td>
                    <td style={{ padding: '7px 10px' }}>{d.attack_duration_h_mean ?? '—'}</td>
                    <td style={{ padding: '7px 10px' }}>{d.il1_blockade_n ?? '—'}</td>
                    <td style={{ padding: '7px 10px' }}>{d.colchicine_response_n ?? '—'}</td>
                    <td style={{ padding: '7px 10px' }}>{d.arthritis_n ?? '—'}</td>
                    <td style={{ padding: '7px 10px' }}>{d.skin_rash_n ?? '—'}</td>
                    <td style={{ padding: '7px 10px', fontSize: 11, maxWidth: 200 }}>{d.pathognomonic || '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {breakdown?.treatment_matrix && (
            <div style={{ marginTop: 24 }}>
              <h3>Treatment Matrix</h3>
              {Object.entries(breakdown.treatment_matrix).map(([gene, tm]) => (
                <div key={gene} style={{ border: `2px solid ${GENE_COLORS[gene] || '#999'}`, borderRadius: 8, padding: 12, marginBottom: 12 }}>
                  <div style={{ fontWeight: 700, color: GENE_COLORS[gene], fontSize: 15, marginBottom: 6 }}>{gene}</div>
                  <div style={{ fontSize: 12 }}><strong>First line:</strong> {tm.first_line}</div>
                  <div style={{ fontSize: 12 }}><strong>Second line:</strong> {tm.second_line}</div>
                  <div style={{ fontSize: 12, color: '#c62828' }}><strong>Avoid:</strong> {tm.avoid}</div>
                  <div style={{ fontSize: 12, color: '#1b5e20' }}><strong>Monitor:</strong> {tm.monitor}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Syndrome Classification Tab */}
      {activeTab === 'Syndrome Classification' && (
        <div>
          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#6a1b9a', color: '#fff' }}>
                  {['Gene', 'Syndrome', 'Attack Duration', 'Trigger', 'Key Lab', 'First-line Rx', 'Amyloid Risk'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {SYNDROME_ROWS.map((row, idx) => (
                  <tr key={row.gene} style={{ background: idx % 2 === 0 ? '#fff' : '#f3e5f5', borderBottom: '1px solid #eee' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] || '#333' }}>{row.gene}</td>
                    <td style={{ padding: '7px 10px', fontWeight: 600 }}>{row.syndrome}</td>
                    <td style={{ padding: '7px 10px' }}>{row.attack_dur}</td>
                    <td style={{ padding: '7px 10px' }}>{row.trigger}</td>
                    <td style={{ padding: '7px 10px' }}>{row.key_lab}</td>
                    <td style={{ padding: '7px 10px' }}>{row.treatment}</td>
                    <td style={{ padding: '7px 10px', color: row.amyloid.startsWith('Yes') ? '#c62828' : '#555', fontWeight: row.amyloid.startsWith('Yes') ? 700 : 400 }}>{row.amyloid}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {breakdown?.syndrome_classification && Object.entries(breakdown.syndrome_classification).map(([group, data]) => (
            <div key={group} style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <h4 style={{ margin: '0 0 8px', color: '#6a1b9a' }}>{group.replace(/_/g, ' ').toUpperCase()}</h4>
              <div style={{ fontSize: 12 }}><strong>Genes:</strong> {data.genes?.join(' · ')}</div>
              <div style={{ fontSize: 12 }}><strong>Mechanism:</strong> {data.mechanism}</div>
              <div style={{ fontSize: 12 }}><strong>Treatment:</strong> {data.treatment}</div>
              {data.biomarker && <div style={{ fontSize: 12 }}><strong>Biomarker:</strong> {data.biomarker}</div>}
              {data.hallmark && <div style={{ fontSize: 12 }}><strong>Hallmark:</strong> {data.hallmark}</div>}
              {data.urgent_note && <div style={{ fontSize: 12, color: '#c62828', fontWeight: 600 }}><strong>URGENT:</strong> {data.urgent_note}</div>}
            </div>
          ))}

          {breakdown?.surveillance_matrix && (
            <div style={{ marginTop: 20 }}>
              <h3>Surveillance Matrix</h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 12 }}>
                {Object.entries(breakdown.surveillance_matrix).map(([gene, items]) => (
                  <div key={gene} style={{ border: `1px solid ${GENE_COLORS[gene] || '#999'}`, borderRadius: 6, padding: 12 }}>
                    <div style={{ fontWeight: 700, color: GENE_COLORS[gene], marginBottom: 6 }}>{gene}</div>
                    {items.map((item, i) => <div key={i} style={{ fontSize: 12, marginBottom: 2 }}>• {item}</div>)}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Definitions Tab */}
      {activeTab === 'Definitions' && defs?.definitions && (
        <div>
          {Object.entries(defs.definitions).map(([term, def]) => (
            <div key={term} style={{ marginBottom: 16, borderLeft: '4px solid #c62828', paddingLeft: 16 }}>
              <h4 style={{ margin: '0 0 6px', color: '#c62828' }}>{term}</h4>
              <p style={{ margin: 0, fontSize: 13, lineHeight: 1.6 }}>{def}</p>
            </div>
          ))}

          {defs?.critical_clinical_rules && (
            <div style={{ background: '#fff3cd', borderRadius: 8, padding: 16, marginTop: 20 }}>
              <h3 style={{ margin: '0 0 12px' }}>Critical Clinical Rules Registry ({Object.keys(defs.critical_clinical_rules).length} rules)</h3>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {Object.keys(defs.critical_clinical_rules).map(rule => (
                  <span key={rule} style={{ background: '#c62828', color: '#fff', padding: '2px 8px', borderRadius: 4, fontSize: 11 }}>{rule}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
