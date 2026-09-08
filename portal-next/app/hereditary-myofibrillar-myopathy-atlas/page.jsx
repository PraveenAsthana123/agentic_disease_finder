'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-myofibrillar-myopathy-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  DES:     '#b71c1c',  // deep red     — MFM1, desmin aggregates, cardiac mandatory
  CRYAB:   '#1565c0',  // deep blue    — MFM2, cataracts pathognomonic, R120G
  MYOT:    '#2e7d32',  // deep green   — MFM3/LGMD1A, late onset, NO cardiac
  FLNC:    '#6a1b9a',  // deep purple  — MFM5, hyaline bodies, truncating→DCM/ARVC
  BAG3:    '#c62828',  // vivid red    — MFM6, childhood onset, DCM mandatory
  PYROXD1: '#e65100',  // deep orange  — AR childhood MFM, hybrid biopsy, no cardiac
  ACTN2:   '#00695c',  // deep teal    — cardiac dominant HCM/DCM/LVNC, Z-disc
  HSPB8:   '#37474f',  // dark grey    — K141N hotspot, distal CMT2L, neuropathy
};

const GENE_INFO = {
  DES:     { full: 'DES / 470aa',     locus: '2q35',    size: '470 aa / 53 kDa',  inh: 'AD/AR', disease: 'MFM1 — CYTOPLASMIC DESMIN AGGREGATES PATHOGNOMONIC / cardiac conduction disease+DCM mandatory / scapuloperoneal+distal / CK 2-10× / onset 20-40yr' },
  CRYAB:   { full: 'CRYAB / 175aa',   locus: '11q23.1', size: '175 aa / 20 kDa',  inh: 'AD/AR', disease: 'MFM2 — POSTERIOR SUBCAPSULAR CATARACTS 50% AD carriers PATHOGNOMONIC / cardiac DCM+HCM / R120G hotspot / small heat shock protein / onset 35-55yr' },
  MYOT:    { full: 'MYOT / 498aa',    locus: '5q31.2',  size: '498 aa / 57 kDa',  inh: 'AD',    disease: 'MFM3/LGMD1A — LATE ONSET 40-65yr PATHOGNOMONIC / NO CARDIAC (key DDx from DES/FLNC/BAG3) / Z-disc filamentous inclusions / dysarthria+dysphagia late' },
  FLNC:    { full: 'FLNC / 2725aa',   locus: '7q32.1',  size: '2725 aa / 291 kDa', inh: 'AD',   disease: 'MFM5 — HYALINE BODIES biopsy PATHOGNOMONIC / ALLELE-SPECIFIC: truncating→DCM/ARVC; missense→MFM5 skeletal / CARDIAC MANDATORY all carriers / most identified MFM gene' },
  BAG3:    { full: 'BAG3 / 575aa',    locus: '10q26.11',size: '575 aa / 62 kDa',  inh: 'AD',    disease: 'MFM6 — CHILDHOOD ONSET 2-15yr PATHOGNOMONIC most severe MFM / DCM MANDATORY childhood / axial hypotonia / respiratory failure early / P209L/P209S hotspot' },
  PYROXD1: { full: 'PYROXD1 / 500aa', locus: '12p12.1', size: '500 aa / 56 kDa',  inh: 'AR',    disease: 'AR childhood MFM — HYBRID BIOPSY (nemaline rods + myofibrillar disruption) / ptosis+facial weakness / N155S/Q372H founders / no cardiac / onset 2-30yr' },
  ACTN2:   { full: 'ACTN2 / 894aa',   locus: '1q43',    size: '894 aa / 104 kDa', inh: 'AD',    disease: 'Alpha-actinin-2 Z-disc — CARDIAC DOMINANT (HCM/DCM/LVNC) / skeletal myopathy mild / mavacamten obstructive HCM 2022 / Z-disc streaming biopsy / CK 1.5-5×' },
  HSPB8:   { full: 'HSPB8 / 196aa',   locus: '12q24.23',size: '196 aa / 22 kDa',  inh: 'AD',    disease: 'HSP22/CMT2L — K141N/K141E HOTSPOT PATHOGNOMONIC / DISTAL ONSET peroneal distribution / rimmed vacuoles + MFM biopsy / axonal neuropathy NCS/EMG mandatory' },
};

function Badge({ text, color }) {
  return (
    <span style={{
      background: color + '22', color, border: `1px solid ${color}55`,
      borderRadius: 4, padding: '2px 7px', fontSize: 11, fontWeight: 700, marginRight: 4,
    }}>{text}</span>
  );
}

function StatCard({ label, value, sub, color }) {
  return (
    <div style={{
      background: '#fff', border: `2px solid ${color || '#e0e0e0'}`,
      borderRadius: 10, padding: '14px 18px', minWidth: 120, textAlign: 'center',
    }}>
      <div style={{ fontSize: 26, fontWeight: 800, color: color || '#333' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 2 }}>{label}</div>
      {sub && <div style={{ fontSize: 11, color: '#888' }}>{sub}</div>}
    </div>
  );
}

function GeneCard({ gene, color, info, data }) {
  return (
    <div style={{
      border: `2px solid ${color}`, borderRadius: 10, padding: 16, marginBottom: 12,
      background: color + '08',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        <span style={{ fontWeight: 800, fontSize: 18, color }}>{gene}</span>
        <Badge text={info.inh} color={color} />
        <Badge text={info.locus} color="#555" />
        <Badge text={info.size} color="#777" />
      </div>
      <div style={{ fontSize: 13, color: '#333', marginBottom: 6 }}>{info.disease}</div>
      {data && (
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12, color: '#555' }}>
          <span>Patients: <b>{data.n_patients}</b></span>
          <span>Onset: <b>{data.mean_age_onset}yr</b></span>
          <span>CK: <b>{data.mean_ck_iu_l} IU/L</b></span>
          <span>Alive: <b>{data.alive_pct}%</b></span>
          <span>Ambulant@10yr: <b>{data.ambulant_10yr_pct}%</b></span>
        </div>
      )}
    </div>
  );
}

export default function HMFMAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  useEffect(() => {
    setLoading(true);
    setErr(null);
    const ep = tab === 'Overview' ? 'overview' : tab === 'Gene Table' ? 'breakdown' : tab === 'Clinical Atlas' ? 'breakdown' : 'definitions';
    const setter = ep === 'overview' ? setOverview : ep === 'breakdown' ? setBreakdown : setDefs;
    const already = ep === 'overview' ? overview : ep === 'breakdown' ? breakdown : defs;
    if (already) { setLoading(false); return; }
    fetch(`${API}/api/${SLUG}/${ep}`)
      .then(r => r.json())
      .then(d => { setter(d); setLoading(false); })
      .catch(e => { setErr(e.message); setLoading(false); });
  }, [tab]);

  const containerStyle = { fontFamily: 'system-ui,sans-serif', maxWidth: 1100, margin: '0 auto', padding: '24px 16px' };
  const tabBarStyle = { display: 'flex', gap: 4, borderBottom: '2px solid #e0e0e0', marginBottom: 24 };
  const tabStyle = (active) => ({
    padding: '8px 18px', border: 'none', borderBottom: active ? '3px solid #b71c1c' : '3px solid transparent',
    background: 'none', cursor: 'pointer', fontWeight: active ? 700 : 400,
    color: active ? '#b71c1c' : '#555', fontSize: 14,
  });

  return (
    <div style={containerStyle}>
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ fontSize: 24, fontWeight: 800, color: '#b71c1c', marginBottom: 6 }}>
          🧬 Hereditary Myofibrillar Myopathy Atlas
        </h1>
        <p style={{ color: '#555', fontSize: 14, margin: 0 }}>
          Complete 8-Gene MFM Spectrum · DES · CRYAB · MYOT · FLNC · BAG3 · PYROXD1 · ACTN2 · HSPB8 · 320 patients · seeds 2206-2213
        </p>
      </div>

      <div style={tabBarStyle}>
        {TABS.map(t => <button key={t} style={tabStyle(tab === t)} onClick={() => setTab(t)}>{t}</button>)}
      </div>

      {loading && <div style={{ color: '#777', padding: 24 }}>Loading…</div>}
      {err && <div style={{ color: 'red', padding: 12 }}>Error: {err}</div>}

      {/* OVERVIEW TAB */}
      {tab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 24 }}>
            <StatCard label="Total Patients" value={overview.n_patients} color="#b71c1c" />
            <StatCard label="Genes Covered" value={overview.n_genes} color="#1565c0" />
            <StatCard label="Alive %" value={`${overview.alive_pct}%`} color="#2e7d32" />
            <StatCard label="Treated %" value={`${overview.treated_pct}%`} color="#6a1b9a" />
            <StatCard label="Mean Onset" value={`${overview.mean_age_onset_yr}yr`} color="#e65100" />
            <StatCard label="Mean Dx Delay" value={`${overview.mean_dx_delay_yr}yr`} color="#c62828" />
            <StatCard label="Mean CK" value={`${overview.mean_ck_iu_l}`} sub="IU/L" color="#00695c" />
          </div>

          <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 12px', color: '#b71c1c' }}>Key Spectrum Facts</h3>
            {overview.key_spectrum_facts?.map((f, i) => (
              <div key={i} style={{ padding: '7px 0', borderBottom: '1px solid #f5f5f5', fontSize: 13, color: '#333' }}>
                <span style={{ marginRight: 8, color: '#b71c1c', fontWeight: 700 }}>▸</span>{f}
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
            <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 10, padding: 16 }}>
              <h4 style={{ margin: '0 0 10px', color: '#e65100' }}>⚠ Cardiac Mandatory Genes</h4>
              {overview.cardiac_mandate_genes?.map(g => (
                <div key={g} style={{ fontSize: 13, color: '#333', padding: '3px 0' }}>
                  <span style={{ fontWeight: 700, color: GENE_COLORS[g] }}>●</span> {g}
                </div>
              ))}
            </div>
            <div style={{ background: '#e8f5e9', border: '2px solid #2e7d32', borderRadius: 10, padding: 16 }}>
              <h4 style={{ margin: '0 0 10px', color: '#2e7d32' }}>✓ No Cardiac (MYOT / PYROXD1 / HSPB8)</h4>
              {overview.no_cardiac_genes?.map(g => (
                <div key={g} style={{ fontSize: 13, color: '#333', padding: '3px 0' }}>
                  <span style={{ fontWeight: 700, color: GENE_COLORS[g] }}>●</span> {g}
                </div>
              ))}
            </div>
          </div>

          <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 20 }}>
            <h3 style={{ margin: '0 0 12px', color: '#555' }}>Critical DDx</h3>
            {overview.critical_ddx && Object.entries(overview.critical_ddx).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 10 }}>
                <span style={{ fontWeight: 700, color: '#b71c1c', fontSize: 12 }}>{k.replace(/_/g, ' ')}</span>
                <div style={{ fontSize: 13, color: '#444', marginTop: 2 }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {tab === 'Gene Table' && breakdown && (
        <div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ background: '#b71c1c', color: '#fff' }}>
                  {['Gene', 'Locus', 'Size', 'Onset (yr)', 'CK IU/L', 'Alive %', 'Ambulant@10yr', 'Dx Delay'].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(breakdown).map(([gene, d], idx) => (
                  <tr key={gene} style={{ background: idx % 2 === 0 ? '#fafafa' : '#fff', borderBottom: '1px solid #f0f0f0' }}>
                    <td style={{ padding: '8px 12px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                    <td style={{ padding: '8px 12px', color: '#555' }}>{d.locus}</td>
                    <td style={{ padding: '8px 12px', color: '#555' }}>{d.protein_size}</td>
                    <td style={{ padding: '8px 12px' }}>{d.mean_age_onset}</td>
                    <td style={{ padding: '8px 12px' }}>{d.mean_ck_iu_l}</td>
                    <td style={{ padding: '8px 12px' }}>{d.alive_pct}%</td>
                    <td style={{ padding: '8px 12px' }}>{d.ambulant_10yr_pct}%</td>
                    <td style={{ padding: '8px 12px' }}>{d.mean_dx_delay_yr}yr</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {tab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.entries(GENE_INFO).map(([gene, info]) => (
            <GeneCard
              key={gene}
              gene={gene}
              color={GENE_COLORS[gene]}
              info={info}
              data={breakdown[gene]}
            />
          ))}

          {/* Per-gene pearls / contraindications */}
          {breakdown && Object.entries(breakdown).map(([gene, d]) => (
            <div key={gene + '_detail'} style={{
              border: `1px solid ${GENE_COLORS[gene]}44`, borderRadius: 8,
              padding: 14, marginBottom: 10, background: '#fafafa',
            }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[gene], marginBottom: 6 }}>
                {gene} — Clinical Pearls & Contraindications
              </div>
              <div style={{ marginBottom: 6 }}>
                <span style={{ fontWeight: 600, fontSize: 12, color: '#555' }}>PEARLS: </span>
                {d.critical_pearls?.map((p, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#333', paddingLeft: 10, paddingBottom: 2 }}>▸ {p}</div>
                ))}
              </div>
              <div>
                <span style={{ fontWeight: 600, fontSize: 12, color: '#c62828' }}>CONTRAINDICATIONS: </span>
                {d.contraindications?.map((c, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#b00020', paddingLeft: 10, paddingBottom: 2 }}>⛔ {c}</div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {tab === 'Definitions' && defs && (
        <div>
          {/* Biopsy pattern table */}
          <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 14px', color: '#b71c1c' }}>Biopsy Pattern Table (IHC / TEM)</h3>
            {defs.biopsy_pattern_table && Object.entries(defs.biopsy_pattern_table).map(([gene, pattern]) => (
              <div key={gene} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #f5f5f5' }}>
                <span style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#555', fontSize: 13 }}>{gene}:</span>
                <span style={{ fontSize: 13, color: '#444', marginLeft: 8 }}>{pattern}</span>
              </div>
            ))}
          </div>

          {/* Cardiac surveillance table */}
          <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 14px', color: '#e65100' }}>Cardiac Surveillance Table</h3>
            {defs.cardiac_surveillance_table && Object.entries(defs.cardiac_surveillance_table).map(([gene, plan]) => (
              <div key={gene} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #ffe0b2' }}>
                <span style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#555', fontSize: 13 }}>{gene}:</span>
                <span style={{ fontSize: 13, color: '#333', marginLeft: 8 }}>{plan}</span>
              </div>
            ))}
          </div>

          {/* MFM IHC panel */}
          {defs.mfm_biopsy_ihc_panel && (
            <div style={{ background: '#e8eaf6', border: '2px solid #3949ab', borderRadius: 10, padding: 20, marginBottom: 20 }}>
              <h3 style={{ margin: '0 0 12px', color: '#3949ab' }}>MFM Biopsy IHC Panel</h3>
              <div style={{ marginBottom: 8 }}>
                <b>First-line:</b> {defs.mfm_biopsy_ihc_panel.first_line?.join(' · ')}
              </div>
              <div style={{ marginBottom: 8 }}>
                <b>Second-line:</b> {defs.mfm_biopsy_ihc_panel.second_line?.join(' · ')}
              </div>
              <div style={{ fontSize: 13, color: '#333', marginTop: 6 }}>{defs.mfm_biopsy_ihc_panel.note}</div>
            </div>
          )}

          {/* Onset age spectrum */}
          <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 12px', color: '#555' }}>Onset Age Spectrum</h3>
            {defs.onset_age_spectrum && Object.entries(defs.onset_age_spectrum).map(([gene, desc]) => (
              <div key={gene} style={{ marginBottom: 8, fontSize: 13 }}>
                <span style={{ fontWeight: 700, color: GENE_COLORS[gene.split('_')[0]] || '#555' }}>{gene}:</span>
                <span style={{ color: '#444', marginLeft: 8 }}>{desc}</span>
              </div>
            ))}
          </div>

          {/* Founder mutations */}
          <div style={{ background: '#f3e5f5', border: '2px solid #7b1fa2', borderRadius: 10, padding: 20, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 12px', color: '#7b1fa2' }}>Founder / Hotspot Mutations</h3>
            {defs.founder_mutations && Object.entries(defs.founder_mutations).map(([mut, desc]) => (
              <div key={mut} style={{ marginBottom: 8, fontSize: 13 }}>
                <span style={{ fontWeight: 700, color: '#7b1fa2' }}>{mut.replace(/_/g, ' ')}:</span>
                <span style={{ color: '#333', marginLeft: 8 }}>{desc}</span>
              </div>
            ))}
          </div>

          {/* Glossary */}
          <div style={{ background: '#fff', border: '1px solid #e0e0e0', borderRadius: 10, padding: 20 }}>
            <h3 style={{ margin: '0 0 12px', color: '#555' }}>Glossary</h3>
            {defs.glossary && Object.entries(defs.glossary).map(([term, def]) => (
              <div key={term} style={{ marginBottom: 10, paddingBottom: 10, borderBottom: '1px solid #f5f5f5' }}>
                <div style={{ fontWeight: 700, fontSize: 13, color: '#333' }}>{term.replace(/_/g, ' ')}</div>
                <div style={{ fontSize: 13, color: '#555', marginTop: 3 }}>{def}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
