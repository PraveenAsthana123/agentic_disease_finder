'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-cmd-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  LAMA2:   '#1565c0',  // deep blue    — MDC1A, absent merosin, leukodystrophy
  COL6A1:  '#2e7d32',  // deep green   — Ullrich/Bethlem, proximal+distal paradox
  COL6A2:  '#388e3c',  // green        — same Ullrich/Bethlem spectrum, β-chain
  COL6A3:  '#43a047',  // mid green    — Bethlem most common, largest subunit
  FKTN:    '#e65100',  // deep orange  — Fukuyama CMD, Japan founder, DCM mandatory
  POMT1:   '#b71c1c',  // deep red     — WWS type 1, most severe, lethal 1yr
  POMT2:   '#c62828',  // vivid red    — WWS type 2, identical to POMT1
  POMGNT1: '#6a1b9a',  // deep purple  — MEB disease, Finnish founder, glaucoma
};

const GENE_INFO = {
  LAMA2:   { full: 'LAMA2 / 3122aa',   locus: '6q22.33',  size: '3122 aa / 395 kDa', inh: 'AR',    disease: 'MDC1A — ABSENT MEROSIN IHC PATHOGNOMONIC / white matter T2 leukodystrophy (non-progressive) / demyelinating neuropathy / most common CMD globally / CK 5-30×' },
  COL6A1:  { full: 'COL6A1 / 1028aa',  locus: '21q22.3',  size: '1028 aa / 140 kDa', inh: 'AD/AR', disease: 'Ullrich CMD/Bethlem — PROXIMAL WEAKNESS + DISTAL HYPERLAXITY PARADOX PATHOGNOMONIC / FOLLICULAR HYPERKERATOSIS+KELOID skin PATHOGNOMONIC / early respiratory failure / CK near-normal' },
  COL6A2:  { full: 'COL6A2 / 1019aa',  locus: '21q22.3',  size: '1019 aa / 140 kDa', inh: 'AR/AD', disease: 'Ullrich CMD/Bethlem — same spectrum as COL6A1 (β-chain) / 21q22.3 same chromosome — cosegregation critical / collagen VI IHC reduced / early respiratory failure Ullrich' },
  COL6A3:  { full: 'COL6A3 / 3177aa',  locus: '2q37.3',   size: '3177 aa / 260 kDa', inh: 'AD/AR', disease: 'Ullrich CMD/Bethlem — most common COL6 gene in Bethlem / largest subunit 3177aa / 2q37.3 different chromosome COL6A1/A2 / keloid+follicular hyperkeratosis Ullrich skin' },
  FKTN:    { full: 'FKTN / 461aa',     locus: '9q31.2',   size: '461 aa / 54 kDa',  inh: 'AR',    disease: 'Fukuyama CMD — INTELLECTUAL DISABILITY MANDATORY / COBBLESTONE LISSENCEPHALY (Type II) PATHOGNOMONIC / DCM adolescence mandatory / c.3036+IVS retrotransposon Japan founder 85%' },
  POMT1:   { full: 'POMT1 / 747aa',    locus: '9q34.13',  size: '747 aa / 83 kDa',  inh: 'AR',    disease: 'Walker-Warburg Syndrome — MOST SEVERE CMD / cobblestone lissencephaly + Z-brainstem + ocular defects PATHOGNOMONIC / LETHAL usually 1st year / POMT1+POMT2 obligate heterodimer' },
  POMT2:   { full: 'POMT2 / 750aa',    locus: '14q24.3',  size: '750 aa / 83 kDa',  inh: 'AR',    disease: 'Walker-Warburg Syndrome type 2 — IDENTICAL to POMT1 WWS (only genetics distinguishes) / POMT1+POMT2 obligate heterodimer / hypomorphic = MEB-like/LGMD-R14 (survival adulthood)' },
  POMGNT1: { full: 'POMGNT1 / 660aa',  locus: '1p34.1',   size: '660 aa / 75 kDa',  inh: 'AR',    disease: 'Muscle-Eye-Brain (MEB) — MYOPIA+GLAUCOMA+CEREBELLAR HYPOPLASIA PATHOGNOMONIC / Finnish founder p.Tyr688Cys 85% / pachygyria / ID mandatory / survival adulthood unlike WWS' },
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
      <div style={{ fontSize: 12, color: '#444', marginBottom: 8 }}>{info.disease}</div>
      {data && (
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginTop: 8 }}>
          <StatCard label="Patients" value={data.n_patients} color={color} />
          <StatCard label="Alive %" value={`${data.alive_pct}%`} color={data.alive_pct > 70 ? '#2e7d32' : '#b71c1c'} />
          <StatCard label="Ambulant@10yr" value={`${data.ambulant_10yr_pct}%`} color={data.ambulant_10yr_pct > 40 ? '#1565c0' : '#e65100'} />
          <StatCard label="Mean CK" value={`${data.mean_ck_iu_l?.toLocaleString()} IU/L`} color="#6a1b9a" />
          <StatCard label="Onset" value={`${data.mean_age_onset} yr`} color={color} />
          <StatCard label="Dx Delay" value={`${data.mean_dx_delay_yr} yr`} color="#555" />
        </div>
      )}
    </div>
  );
}

export default function HereditoryCMDAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [ov, setOv] = useState(null);
  const [br, setBr] = useState(null);
  const [defs, setDefs] = useState(null);
  const [err, setErr] = useState('');

  useEffect(() => {
    fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()).then(setOv).catch(() => setErr('Overview failed'));
    fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()).then(setBr).catch(() => setErr('Breakdown failed'));
    fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()).then(setDefs).catch(() => setErr('Definitions failed'));
  }, []);

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: 24, fontFamily: 'system-ui, sans-serif' }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1565c0,#b71c1c)', borderRadius: 14, padding: '24px 28px', color: '#fff', marginBottom: 24 }}>
        <div style={{ fontSize: 13, opacity: 0.85, marginBottom: 6 }}>&#x1f9ec; Hereditary Disease Atlas</div>
        <h1 style={{ margin: 0, fontSize: 24, fontWeight: 800 }}>Hereditary CMD Atlas</h1>
        <div style={{ fontSize: 13, opacity: 0.9, marginTop: 6 }}>
          Complete 8-Gene Congenital Muscular Dystrophy (CMD) Spectrum &mdash;
          LAMA2 &middot; COL6A1 &middot; COL6A2 &middot; COL6A3 &middot; FKTN &middot; POMT1 &middot; POMT2 &middot; POMGNT1
        </div>
        <div style={{ fontSize: 11, opacity: 0.75, marginTop: 4 }}>
          320-patient aggregate &bull; seeds 2214-2221 &bull; MDC1A / Ullrich / Bethlem / Fukuyama / WWS / MEB spectrum
        </div>
      </div>

      {err && <div style={{ color: 'red', marginBottom: 12 }}>⚠️ {err}</div>}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 6, marginBottom: 20, flexWrap: 'wrap' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            padding: '8px 18px', borderRadius: 8, border: 'none', cursor: 'pointer',
            background: tab === t ? '#1565c0' : '#e8edf3',
            color: tab === t ? '#fff' : '#333', fontWeight: tab === t ? 700 : 400, fontSize: 13,
          }}>{t}</button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'Overview' && ov && (
        <div>
          {/* Aggregate stats */}
          <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', marginBottom: 20 }}>
            <StatCard label="Genes" value={ov.n_genes} color="#1565c0" />
            <StatCard label="Patients" value={ov.n_patients} color="#333" />
            <StatCard label="Alive %" value={`${ov.alive_pct}%`} color="#2e7d32" />
            <StatCard label="Treated %" value={`${ov.treated_pct}%`} color="#e65100" />
            <StatCard label="Ambulant@10yr" value={`${ov.ambulant_10yr_pct}%`} color="#1565c0" />
            <StatCard label="Mean Onset" value={`${ov.mean_age_onset_yr} yr`} color="#555" />
            <StatCard label="Mean Dx Delay" value={`${ov.mean_dx_delay_yr} yr`} color="#777" />
            <StatCard label="Mean CK" value={`${ov.mean_ck_iu_l?.toLocaleString()} IU/L`} color="#6a1b9a" />
          </div>

          {/* Key facts */}
          <div style={{ background: '#f8f9fa', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            <h3 style={{ margin: '0 0 12px', color: '#1565c0' }}>Key Spectrum Facts</h3>
            {(ov.key_spectrum_facts || []).map((f, i) => (
              <div key={i} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <span style={{ color: '#1565c0', fontWeight: 700, minWidth: 18 }}>▶</span>
                <span style={{ fontSize: 13, color: '#333' }}>{f}</span>
              </div>
            ))}
          </div>

          {/* Gene group cards */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 20 }}>
            <div style={{ background: '#e8f5e9', borderRadius: 10, padding: 14 }}>
              <div style={{ fontWeight: 700, color: '#2e7d32', marginBottom: 8 }}>Collagen VI Genes (COL6)</div>
              {(ov.collagen_vi_genes || []).map(g => (
                <div key={g} style={{ fontSize: 12, color: '#333', marginBottom: 4 }}>
                  <strong style={{ color: GENE_COLORS[g] }}>{g}</strong> — {GENE_INFO[g]?.locus}
                </div>
              ))}
              <div style={{ fontSize: 11, color: '#555', marginTop: 8 }}>Ullrich CMD (severe AR) / Bethlem Myopathy (mild AD) — CK near-normal; collagen VI IHC reduced</div>
            </div>
            <div style={{ background: '#fff3e0', borderRadius: 10, padding: 14 }}>
              <div style={{ fontWeight: 700, color: '#e65100', marginBottom: 8 }}>Dystroglycanopathy Genes (αDG)</div>
              {(ov.dystroglycanopathy_genes || []).map(g => (
                <div key={g} style={{ fontSize: 12, color: '#333', marginBottom: 4 }}>
                  <strong style={{ color: GENE_COLORS[g] }}>{g}</strong> — {GENE_INFO[g]?.locus}
                </div>
              ))}
              <div style={{ fontSize: 11, color: '#555', marginTop: 8 }}>IIH6 α-dystroglycan IHC reduced in all — gene panel required to distinguish</div>
            </div>
            <div style={{ background: '#e3f2fd', borderRadius: 10, padding: 14 }}>
              <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 8 }}>No Intellectual Disability</div>
              {(ov.no_intellectual_disability_genes || []).map(g => (
                <span key={g} style={{ fontSize: 12, color: GENE_COLORS[g], fontWeight: 700, marginRight: 8 }}>{g}</span>
              ))}
              <div style={{ fontSize: 11, color: '#555', marginTop: 8 }}>LAMA2, COL6A1/A2/A3 — cognition preserved unless epilepsy complications</div>
            </div>
            <div style={{ background: '#fce4ec', borderRadius: 10, padding: 14 }}>
              <div style={{ fontWeight: 700, color: '#b71c1c', marginBottom: 8 }}>Lethal (Usually 1st Year)</div>
              {(ov.lethal_usually_1yr_genes || []).map(g => (
                <span key={g} style={{ fontSize: 12, color: GENE_COLORS[g], fontWeight: 700, marginRight: 8 }}>{g}</span>
              ))}
              <div style={{ fontSize: 11, color: '#555', marginTop: 8 }}>Walker-Warburg Syndrome — brainstem failure; palliative goals-of-care discussion mandatory</div>
            </div>
          </div>

          {/* Critical DDx */}
          <div style={{ background: '#fff8e1', borderRadius: 10, padding: 16 }}>
            <h3 style={{ margin: '0 0 12px', color: '#e65100' }}>Critical DDx Pearls</h3>
            {Object.entries(ov.critical_ddx || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <span style={{ color: '#e65100', fontWeight: 700, minWidth: 18 }}>⚡</span>
                <span style={{ fontSize: 12, color: '#333' }}><strong>{k.replace(/_/g,' ')}:</strong> {v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Gene Table Tab ── */}
      {tab === 'Gene Table' && br && (
        <div>
          <h3 style={{ color: '#1565c0' }}>Per-Gene Clinical Summary (8 Genes, 40 patients each)</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene','Locus','Protein','Inheritance','n','Alive%','Ambulant@10yr%','Mean CK (IU/L)','Mean Onset (yr)','Dx Delay (yr)'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(br).map(([gene, d], i) => (
                  <tr key={gene} style={{ background: i % 2 === 0 ? '#f5f8ff' : '#fff' }}>
                    <td style={{ padding: '7px 10px', fontWeight: 700, color: GENE_COLORS[gene] }}>{gene}</td>
                    <td style={{ padding: '7px 10px' }}>{d.locus}</td>
                    <td style={{ padding: '7px 10px' }}>{d.protein_size}</td>
                    <td style={{ padding: '7px 10px' }}>{d.inheritance?.split(';')[0]}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{d.n_patients}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center', color: d.alive_pct > 70 ? '#2e7d32' : '#b71c1c', fontWeight: 700 }}>{d.alive_pct}%</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center', color: d.ambulant_10yr_pct > 30 ? '#1565c0' : '#e65100', fontWeight: 700 }}>{d.ambulant_10yr_pct}%</td>
                    <td style={{ padding: '7px 10px', textAlign: 'right' }}>{d.mean_ck_iu_l?.toLocaleString()}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{d.mean_age_onset}</td>
                    <td style={{ padding: '7px 10px', textAlign: 'center' }}>{d.mean_dx_delay_yr}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Key features per gene */}
          <h3 style={{ color: '#1565c0', marginTop: 24 }}>Key Clinical Features per Gene</h3>
          {Object.entries(br).map(([gene, d]) => (
            <div key={gene} style={{ marginBottom: 18, border: `1px solid ${GENE_COLORS[gene]}44`, borderRadius: 8, padding: 14 }}>
              <div style={{ fontWeight: 800, color: GENE_COLORS[gene], marginBottom: 8, fontSize: 15 }}>{gene}</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 10 }}>
                <div>
                  <div style={{ fontSize: 11, color: '#777', fontWeight: 600 }}>KEY FEATURES</div>
                  {(d.key_features || []).map((f, i) => (
                    <div key={i} style={{ fontSize: 12, color: '#333', marginTop: 3 }}>• {f}</div>
                  ))}
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#777', fontWeight: 600 }}>CONTRAINDICATIONS</div>
                  <div style={{ fontSize: 12, color: '#b71c1c', marginTop: 3 }}>{d.contraindications}</div>
                </div>
              </div>
              <div>
                <div style={{ fontSize: 11, color: '#777', fontWeight: 600 }}>CRITICAL PEARLS</div>
                {(d.critical_pearls || []).map((p, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#444', marginTop: 3 }}>⚡ {p}</div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── Clinical Atlas Tab ── */}
      {tab === 'Clinical Atlas' && br && (
        <div>
          <h3 style={{ color: '#1565c0' }}>Clinical Atlas — Per-Gene Profiles</h3>
          {Object.keys(GENE_INFO).map(gene => (
            <GeneCard key={gene} gene={gene} color={GENE_COLORS[gene]} info={GENE_INFO[gene]} data={br[gene]} />
          ))}
        </div>
      )}

      {/* ── Definitions Tab ── */}
      {tab === 'Definitions' && defs && (
        <div>
          <h3 style={{ color: '#1565c0' }}>CMD Severity Spectrum</h3>
          <div style={{ background: '#f8f9fa', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            {Object.entries(defs.cmd_severity_spectrum || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <span style={{ fontWeight: 700, color: '#1565c0', minWidth: 100, fontSize: 12 }}>{k.replace(/_/g,' ')}</span>
                <span style={{ fontSize: 12, color: '#444' }}>{v}</span>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#2e7d32' }}>IHC Marker Table</h3>
          <div style={{ background: '#f1f8e9', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            {Object.entries(defs.ihc_marker_table || {}).map(([gene, v]) => (
              <div key={gene} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <span style={{ fontWeight: 700, color: GENE_COLORS[gene], minWidth: 80, fontSize: 12 }}>{gene}</span>
                <span style={{ fontSize: 12, color: '#333' }}>{v}</span>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#6a1b9a' }}>Brain MRI Table</h3>
          <div style={{ background: '#f3e5f5', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            {Object.entries(defs.brain_mri_table || {}).map(([gene, v]) => (
              <div key={gene} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <span style={{ fontWeight: 700, color: GENE_COLORS[gene], minWidth: 80, fontSize: 12 }}>{gene}</span>
                <span style={{ fontSize: 12, color: '#333' }}>{v}</span>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#e65100' }}>Ocular Involvement</h3>
          <div style={{ background: '#fff8e1', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            {Object.entries(defs.ocular_involvement_table || {}).map(([gene, v]) => (
              <div key={gene} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <span style={{ fontWeight: 700, color: GENE_COLORS[gene], minWidth: 80, fontSize: 12 }}>{gene}</span>
                <span style={{ fontSize: 12, color: '#333' }}>{v}</span>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#b71c1c' }}>Founder Mutations</h3>
          <div style={{ background: '#ffebee', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            {Object.entries(defs.founder_mutations || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <span style={{ fontWeight: 700, color: '#b71c1c', minWidth: 180, fontSize: 12 }}>{k.replace(/_/g,' ')}</span>
                <span style={{ fontSize: 12, color: '#333' }}>{v}</span>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#333' }}>Dystroglycanopathy Spectrum</h3>
          <div style={{ background: '#f5f5f5', borderRadius: 10, padding: 16, marginBottom: 20 }}>
            {Object.entries(defs.dystroglycanopathy_spectrum || {}).map(([k, v]) => (
              <div key={k} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <span style={{ fontWeight: 700, color: '#555', minWidth: 120, fontSize: 12 }}>{k.replace(/_/g,' ')}</span>
                <span style={{ fontSize: 12, color: '#333' }}>{v}</span>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#333' }}>Gene Definitions</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
            {Object.entries(defs.gene_definitions || {}).map(([gene, d]) => (
              <div key={gene} style={{ border: `1px solid ${GENE_COLORS[gene]}55`, borderRadius: 8, padding: 12 }}>
                <div style={{ fontWeight: 800, color: GENE_COLORS[gene], marginBottom: 6 }}>{gene}</div>
                <div style={{ fontSize: 11, color: '#555' }}><strong>Locus:</strong> {d.locus}</div>
                <div style={{ fontSize: 11, color: '#555' }}><strong>Size:</strong> {d.protein_size}</div>
                <div style={{ fontSize: 11, color: '#444', marginTop: 4 }}>{d.inheritance_detail?.substring(0, 200)}...</div>
              </div>
            ))}
          </div>

          <h3 style={{ color: '#333' }}>Glossary</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
            {Object.entries(defs.glossary || {}).map(([k, v]) => (
              <div key={k} style={{ background: '#f8f9fa', borderRadius: 6, padding: '8px 12px' }}>
                <div style={{ fontWeight: 700, color: '#1565c0', fontSize: 12, marginBottom: 3 }}>{k.replace(/_/g,' ')}</div>
                <div style={{ fontSize: 11, color: '#555' }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Loading state */}
      {!ov && !err && (
        <div style={{ textAlign: 'center', color: '#888', padding: 40 }}>Loading CMD atlas data…</div>
      )}
    </div>
  );
}
