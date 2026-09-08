'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-chorea-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  HTT:    '#1a237e',  // deep indigo    — Huntington disease CAG repeat
  VPS13A: '#880e4f',  // deep pink      — Chorea-acanthocytosis acanthocytes
  PANK2:  '#bf360c',  // deep burnt     — PKAN eye-of-tiger NBIA1
  WDR45:  '#4a148c',  // deep purple    — BPAN biphasic NBIA5 XL
  FTL:    '#006064',  // deep teal      — Neuroferritinopathy low ferritin paradox
  'NKX2-1': '#1b5e20', // deep green   — Benign hereditary chorea brain-lung-thyroid
  JPH3:   '#e65100',  // deep orange    — HDL2 African ancestry JPH3
  TBP:    '#37474f',  // dark slate     — SCA17/HDL4 ataxia+chorea
};

const GENE_INFO = {
  HTT:    { full: 'HTT / Huntingtin 3144aa PolyQ',       locus: '4p16.3',   size: '3144 aa', inh: 'AD', disease: 'Huntington Disease — CAG≥36 / CAG≥60 Juvenile / Tetrabenazine FDA / Caudate Atrophy' },
  VPS13A: { full: 'VPS13A / Vacuolar Protein Sorting 13A', locus: '9q21.2', size: '3174 aa', inh: 'AR', disease: 'Chorea-Acanthocytosis (ChAc) — Acanthocytes PATHOGNOMONIC / Elevated CK / Tongue-Lip Biting' },
  PANK2:  { full: 'PANK2 / Pantothenate Kinase 2',       locus: '20p13',    size: '570 aa',  inh: 'AR', disease: 'PKAN-NBIA1 — Eye-of-Tiger T2 MRI PATHOGNOMONIC / GPi Iron / Pantethine' },
  WDR45:  { full: 'WDR45 / WIPI4 β-propeller',           locus: 'Xp11.23',  size: '330 aa',  inh: 'XL', disease: 'BPAN-NBIA5 — Biphasic: Epilepsy+ID → Parkinsonism+Dementia / De Novo / Females' },
  FTL:    { full: 'FTL / Ferritin Light Chain 24-mer',   locus: '19q13.33', size: '175 aa',  inh: 'AD', disease: 'Neuroferritinopathy — LOW Ferritin PARADOX / Cysts T2 MRI / NO Iron Supplement' },
  'NKX2-1': { full: 'NKX2-1 / TTF-1 NK2 Homeobox',     locus: '14q13.3',  size: '401 aa',  inh: 'AD', disease: 'Benign Hereditary Chorea — Brain-Lung-Thyroid / NOT Progressive / Levodopa Response' },
  JPH3:   { full: 'JPH3 / Junctophilin 3',               locus: '16q24.2',  size: '741 aa',  inh: 'AD', disease: 'Huntington Disease-Like 2 (HDL2) — African Ancestry / CTG≥41 / Test HTT First' },
  TBP:    { full: 'TBP / TATA-Binding Protein PolyQ',    locus: '6q27',     size: '339 aa',  inh: 'AD', disease: 'SCA17/HDL4 — CAG≥49 / Ataxia+Chorea KEY Distinguisher / Cerebellar Atrophy MRI' },
};

const FLAG_BADGE = ({ flag }) => {
  const bg = flag.includes('PATHOGNOMONIC') ? '#b71c1c'
    : flag.includes('MANDATORY') || flag.includes('FDA') || flag.includes('APPROVED') ? '#1565c0'
    : flag.includes('CI') || flag.includes('CONTRAINDICATED') || flag.includes('DO-NOT') || flag.includes('AVOID') ? '#880e4f'
    : flag.includes('EMERGENCY') ? '#e65100'
    : flag.includes('DISTINGUISH') || flag.includes('DDx') || flag.includes('FIRST') ? '#2e7d32'
    : '#37474f';
  return (
    <span style={{
      background: bg, color: '#fff', borderRadius: 4,
      padding: '2px 7px', fontSize: 11, margin: '2px 3px', display: 'inline-block',
    }}>{flag}</span>
  );
};

export default function HereditaryChoreaAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, br, def]) => { setOverview(ov); setBreakdown(br); setDefinitions(def); })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: 24, maxWidth: 1200, margin: '0 auto' }}>
      <h1 style={{ color: '#1a237e', marginBottom: 4 }}>
        🕺 Hereditary Chorea Atlas
      </h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Atlas — HTT (HD) · VPS13A (ChAc) · PANK2 (PKAN/NBIA1) · WDR45 (BPAN/NBIA5) · FTL (Neuroferritinopathy) · NKX2-1 (BHC) · JPH3 (HDL2) · TBP (SCA17/HDL4)
        &nbsp;|&nbsp; 320 patients · seeds 2078-2085
      </p>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 24 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)} style={{
            padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
            background: activeTab === t ? '#1a237e' : '#e8eaf6',
            color: activeTab === t ? '#fff' : '#1a237e', fontWeight: activeTab === t ? 700 : 400,
          }}>{t}</button>
        ))}
      </div>

      {loading && <p style={{ color: '#888' }}>Loading…</p>}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}

      {/* OVERVIEW TAB */}
      {activeTab === 'Overview' && overview && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(210px, 1fr))', gap: 14, marginBottom: 28 }}>
            {[
              { label: 'Total Patients', value: overview.total_patients, color: '#1a237e' },
              { label: 'Tetrabenazine Patients (HTT)', value: overview.tetrabenazine_patients, color: '#1565c0' },
              { label: 'Psychiatric First (HTT/JPH3)', value: overview.psychiatric_first_patients, color: '#4a148c' },
              { label: 'Acanthocytes (VPS13A/JPH3)', value: overview.acanthocytes_patients, color: '#880e4f' },
              { label: 'Eye-of-Tiger MRI (PANK2)', value: overview.eye_tiger_mri_patients, color: '#bf360c' },
              { label: 'Biphasic BPAN (WDR45)', value: overview.biphasic_bpan_patients, color: '#4a148c' },
              { label: 'Low Ferritin Paradox (FTL)', value: overview.low_ferritin_paradox_patients, color: '#006064' },
              { label: 'Thyroid Dysfunction (NKX2-1)', value: overview.thyroid_dysfunction_bhc_patients, color: '#1b5e20' },
              { label: 'African Ancestry (JPH3)', value: overview.african_ancestry_hdl2_patients, color: '#e65100' },
              { label: 'Cerebellar Atrophy (TBP)', value: overview.cerebellar_atrophy_sca17_patients, color: '#37474f' },
              { label: 'Tongue Biting (VPS13A)', value: overview.tongue_biting_chac_patients, color: '#880e4f' },
              { label: 'Seeds', value: overview.seeds, color: '#455a64', isText: true },
            ].map(({ label, value, color, isText }) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: '14px 18px', borderLeft: `4px solid ${color}` }}>
                <div style={{ fontSize: 12, color: '#777', marginBottom: 4 }}>{label}</div>
                <div style={{ fontSize: isText ? 16 : 28, fontWeight: 700, color }}>{value}</div>
              </div>
            ))}
          </div>

          {/* Gene colour legend */}
          <h3 style={{ color: '#1a237e', marginBottom: 12 }}>8-Gene Hereditary Chorea Spectrum</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 20 }}>
            {Object.entries(GENE_INFO).map(([gene, info]) => (
              <div key={gene} style={{
                background: GENE_COLORS[gene], color: '#fff',
                borderRadius: 6, padding: '8px 14px', minWidth: 200,
              }}>
                <div style={{ fontWeight: 700, fontSize: 15 }}>{gene}</div>
                <div style={{ fontSize: 11, opacity: 0.88 }}>{info.locus} · {info.size} · {info.inh}</div>
                <div style={{ fontSize: 11, opacity: 0.80, marginTop: 2 }}>{info.disease}</div>
              </div>
            ))}
          </div>

          {/* Key clinical rules */}
          <h3 style={{ color: '#1a237e', marginBottom: 8 }}>Key Clinical Rules — Hereditary Chorea</h3>
          <div style={{ background: '#e8eaf6', borderRadius: 8, padding: 16 }}>
            <ul style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8 }}>
              <li><strong>TETRABENAZINE/DEUTETRABENAZINE FOR HD CHOREA ONLY</strong> — VMAT2 inhibitors are HTT-approved; contraindicated in primary dystonia (TOR1A/THAP1); do not use in other choreic syndromes without care</li>
              <li><strong>HD PRESYMPTOMATIC TESTING = MANDATORY COUNSELLING PROTOCOL</strong> — minimum 2 counselling sessions before testing; result in person with counsellor; applies to HTT, JPH3, TBP</li>
              <li><strong>ACANTHOCYTES: FRESH HEPARINISED BLOOD ONLY</strong> — EDTA causes artefactual acanthocytosis; always repeat on fresh smear before concluding positive</li>
              <li><strong>EYE-OF-THE-TIGER (PANK2) = PATHOGNOMONIC</strong> — bilateral GPi T2 hypointensity + central hyperintensity; no other NBIA has this exact pattern</li>
              <li><strong>FTL LOW FERRITIN = PARADOX</strong> — DO NOT supplement iron; paradoxically LOW serum ferritin despite brain iron overload; prescribing iron is harmful</li>
              <li><strong>NKX2-1 (BHC) IS NOT PROGRESSIVE</strong> — unlike HD; thyroid function tests annually lifelong; levodopa trial often dramatically effective</li>
              <li><strong>TEST HTT FIRST, THEN JPH3 (AFRICAN ANCESTRY), THEN TBP</strong> — HD phenocopy algorithm; JPH3 almost exclusively African ancestry; TBP distinguished by ataxia</li>
              <li><strong>BPAN (WDR45) BIPHASIC COURSE</strong> — Phase 1: childhood epilepsy+ID; Phase 2: adult parkinsonism+dementia; DaTscan abnormal in phase 2; de novo, predominantly females</li>
              <li><strong>VPS13A: ELEVATED CK + ACANTHOCYTES</strong> — both PATHOGNOMONIC; distinguish McLeod syndrome (XK, X-linked, Kell antigen weak) vs ChAc (AR, normal Kell)</li>
              <li><strong>TBP (SCA17): ATAXIA + CHOREA</strong> — cerebellar involvement distinguishes from pure HD; cerebellar atrophy on MRI</li>
            </ul>
          </div>
        </div>
      )}

      {/* GENE TABLE TAB */}
      {activeTab === 'Gene Table' && breakdown && (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#1a237e', color: '#fff' }}>
                {['Gene', 'Locus', 'Size', 'Inh.', 'Patients', 'Disease / Syndrome', 'Key Treatment', 'Critical Flags'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(breakdown).map(([gene, data], i) => (
                <tr key={gene} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                  <td style={{ padding: '9px 12px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                  <td style={{ padding: '9px 12px' }}>{data.locus}</td>
                  <td style={{ padding: '9px 12px' }}>{data.protein_size}</td>
                  <td style={{ padding: '9px 12px' }}>{data.inheritance.split(';')[0].split('(')[0].trim()}</td>
                  <td style={{ padding: '9px 12px', textAlign: 'center', fontWeight: 700 }}>{data.patient_count}</td>
                  <td style={{ padding: '9px 12px', maxWidth: 220, fontSize: 12 }}>{GENE_INFO[gene]?.disease || '—'}</td>
                  <td style={{ padding: '9px 12px', maxWidth: 200, fontSize: 12 }}>{data.treatment?.split(';')[0]?.replace(/\*\*/g, '') || '—'}</td>
                  <td style={{ padding: '9px 12px', maxWidth: 260 }}>
                    <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                      {(data.critical_flags || []).slice(0, 3).map(f => <FLAG_BADGE key={f} flag={f} />)}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* CLINICAL ATLAS TAB */}
      {activeTab === 'Clinical Atlas' && breakdown && (
        <div>
          {Object.entries(breakdown).map(([gene, data]) => (
            <div key={gene} style={{
              borderLeft: `5px solid ${GENE_COLORS[gene]}`,
              background: '#fafafa', borderRadius: 8, padding: 18, marginBottom: 20,
            }}>
              <h3 style={{ color: GENE_COLORS[gene], marginTop: 0, marginBottom: 6 }}>
                {gene} — {GENE_INFO[gene]?.full}
              </h3>
              <div style={{ fontSize: 12, color: '#777', marginBottom: 10 }}>
                {data.locus} · {data.protein_size} · {data.inheritance?.split(';')[0]}
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                <div>
                  <div style={{ fontWeight: 600, color: '#444', marginBottom: 4 }}>Age of Onset</div>
                  <div style={{ fontSize: 13, color: '#555' }}>{data.age_of_onset}</div>
                </div>
                <div>
                  <div style={{ fontWeight: 600, color: '#444', marginBottom: 4 }}>Key Biomarker</div>
                  <div style={{ fontSize: 13, color: '#555' }}>{data.key_biomarker}</div>
                </div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, color: '#b71c1c', marginBottom: 4 }}>Pathognomonic Signs</div>
                <div style={{ fontSize: 13, color: '#555' }}>{data.pathognomonic}</div>
              </div>

              <div style={{ marginBottom: 10 }}>
                <div style={{ fontWeight: 600, color: '#1565c0', marginBottom: 4 }}>Treatment</div>
                <div style={{ fontSize: 13, color: '#555' }}>{data.treatment}</div>
              </div>

              <div>
                <div style={{ fontWeight: 600, color: '#444', marginBottom: 6 }}>Critical Flags</div>
                <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                  {(data.critical_flags || []).map(f => <FLAG_BADGE key={f} flag={f} />)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* DEFINITIONS TAB */}
      {activeTab === 'Definitions' && definitions && (
        <div>
          <h3 style={{ color: '#1a237e', marginBottom: 12 }}>Gene Definitions</h3>
          {Object.entries(definitions.genes || {}).map(([gene, desc]) => (
            <div key={gene} style={{ marginBottom: 14, borderLeft: `4px solid ${GENE_COLORS[gene] || '#455a64'}`, paddingLeft: 14 }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#455a64', marginBottom: 2 }}>{gene}</div>
              <div style={{ fontSize: 13, color: '#555', lineHeight: 1.6 }}>{desc.replace(/--/g, '·')}</div>
            </div>
          ))}

          <h3 style={{ color: '#1a237e', marginTop: 28, marginBottom: 12 }}>Glossary</h3>
          {Object.entries(definitions.glossary || {}).map(([term, def]) => (
            <div key={term} style={{ marginBottom: 12, background: '#f5f5f5', borderRadius: 6, padding: '10px 14px' }}>
              <div style={{ fontWeight: 600, color: '#1a237e', marginBottom: 3 }}>{term}</div>
              <div style={{ fontSize: 13, color: '#555', lineHeight: 1.6 }}>{def}</div>
            </div>
          ))}

          <h3 style={{ color: '#1a237e', marginTop: 28, marginBottom: 12 }}>Surveillance Protocols</h3>
          {Object.entries(definitions.surveillance_protocols || {}).map(([gene, protocol]) => (
            <div key={gene} style={{ marginBottom: 12, borderLeft: `4px solid ${GENE_COLORS[gene.split(' ')[0]] || '#455a64'}`, paddingLeft: 14 }}>
              <div style={{ fontWeight: 700, color: GENE_COLORS[gene.split(' ')[0]] || '#455a64', marginBottom: 3 }}>{gene}</div>
              <div style={{ fontSize: 13, color: '#555', lineHeight: 1.6 }}>{protocol}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
