'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-transthyretin-amyloidosis-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'TTR':   '#1a237e',  // deep navy       -- ATTR; V30M; V122I; Tafamidis/Patisiran; DPD-PATHOGNOMONIC
  'APOA1': '#4a148c',  // deep purple      -- AApoAI; HDL-near-zero-PATHOGNOMONIC; multi-organ
  'FGA':   '#1b5e20',  // dark green       -- AFib; renal-dominant; Liver-Tx-CURATIVE
  'LYZ':   '#0d47a1',  // medium blue      -- ALys; hepatomegaly+splenomegaly-PATHOGNOMONIC
  'GSN':   '#880e4f',  // deep crimson     -- AGel; MERETOJA-TRIAD-PATHOGNOMONIC; Finnish
  'APOA2': '#e65100',  // burnt orange     -- AApoAII; renal; Ter94-stop-codon
  'B2M':   '#263238',  // dark slate       -- AB2M; dialysis-CTS-PATHOGNOMONIC; HDF
  'OSMR':  '#4e342e',  // dark brown       -- PLCA; pruritic-interscapular; NO-systemic
};

const GENE_INFO = {
  'TTR':   { full: 'Transthyretin / ATTR / V30M-neuropathy / V122I-cardiac-African-American / Tafamidis-Patisiran-Vutrisiran-FDA', locus: '18q12.1', size: '127 aa / 14 kDa monomer', inh: 'AD amyloidogenic' },
  'APOA1': { full: 'ApoA-I / AApoAI / multi-organ-renal-hepatic-cardiac / HDL-near-zero-PATHOGNOMONIC', locus: '11q23.3', size: '267 aa / 28 kDa', inh: 'AD LOF amyloidogenic' },
  'FGA':   { full: 'Fibrinogen-alpha / AFib / renal-dominant-Glu526Val / NO-neuropathy / Liver-Tx-CURATIVE', locus: '4q28.1', size: '866 aa / 95 kDa', inh: 'AD LOF amyloidogenic' },
  'LYZ':   { full: 'Lysozyme / ALys / non-neuropathic / hepatomegaly+splenomegaly-PATHOGNOMONIC / D67H-I56T-UK', locus: '12q15', size: '148 aa / 14 kDa', inh: 'AD GOF amyloidogenic' },
  'GSN':   { full: 'Gelsolin / AGel / MERETOJA-TRIAD / lattice-corneal-II+facial-VII+cutis-laxa / D187N-Finnish', locus: '9q33.2', size: '782 aa / 80 kDa', inh: 'AD GOF amyloidogenic' },
  'APOA2': { full: 'ApoA-II / AApoAII / renal-dominant / stop-codon-Ter94 / late-onset-50-70yr', locus: '1q23.3', size: '100 aa / 8.7 kDa', inh: 'AD LOF amyloidogenic' },
  'B2M':   { full: 'Beta-2 microglobulin / AB2M / dialysis-related-DRA / carpal-tunnel-PATHOGNOMONIC / D76N-hereditary-rare', locus: '15q21.1', size: '119 aa / 11.8 kDa', inh: 'AR rare hereditary / acquired dialysis' },
  'OSMR':  { full: 'Oncostatin-M receptor beta / PLCA / pruritic-papules-interscapular-PATHOGNOMONIC / NO-systemic-organs', locus: '5p13.1', size: '979 aa / 110 kDa', inh: 'AD GOF' },
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
        <Badge text={`Cardiac: ${cohort.cardiac_n}/40`} color='#b71c1c' />
        <Badge text={`Neuro: ${cohort.neuropathy_n}/40`} color='#6a1b9a' />
        <Badge text={`Renal: ${cohort.renal_n}/40`} color='#0277bd' />
      </div>
      {cohort.corneal_lattice_n > 0 && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          👁 <b>Corneal Lattice II:</b> {cohort.corneal_lattice_n}/40 — TYPE II (NOT Type I) = GSN/MERETOJA
        </div>
      )}
      {cohort.carpal_tunnel_n > 0 && (
        <div style={{ fontSize: 12, background: '#e3f2fd', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🤝 <b>Carpal Tunnel:</b> {cohort.carpal_tunnel_n}/40 — bilateral CTS in dialysis patient PATHOGNOMONIC AB2M
        </div>
      )}
      {cohort.cutaneous_n > 0 && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🩹 <b>Pruritic Papules:</b> {cohort.cutaneous_n}/40 — interscapular PATHOGNOMONIC PLCA
        </div>
      )}
      {!cohort.systemic_organs && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#880e4f' }}>
          ✅ NO systemic organ involvement — PLCA/OSMR only: distinguishes from all other amyloidoses
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
      <div style={{ background: '#1a237e', color: '#fff', borderRadius: 8, padding: 20, marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22 }}>🧬 Hereditary Transthyretin Amyloidosis Atlas</h2>
        <p style={{ margin: '8px 0 0', opacity: 0.9, fontSize: 13 }}>{data.subtitle}</p>
        <div style={{ marginTop: 12, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <Badge text={`${data.total_patients} Patients`} color='#0d47a1' />
          <Badge text={`8 Genes`} color='#4a148c' />
          <Badge text={`Seeds ${data.seeds}`} color='#1b5e20' />
        </div>
      </div>
      <div style={{ background: '#fff3e0', border: '2px solid #ff6f00', borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <b>🚨 Key Clinical Rules:</b>
        <ul style={{ margin: '8px 0 0', paddingLeft: 20, fontSize: 13 }}>
          <li><b>TTR cardiac:</b> Apical-sparing strain + 99mTc-DPD Grade 2-3 = PATHOGNOMONIC — AVOID digoxin/verapamil</li>
          <li><b>FGA:</b> Liver transplant CURATIVE — renal-dominant, NO neuropathy</li>
          <li><b>GSN:</b> MERETOJA TRIAD (corneal lattice II + facial VII + cutis laxa) = PATHOGNOMONIC</li>
          <li><b>B2M:</b> Bilateral CTS in dialysis patient &gt;5-10yr = PATHOGNOMONIC</li>
          <li><b>OSMR:</b> NO systemic organs — pruritic interscapular papules only</li>
          <li><b>ALL:</b> Exclude AL amyloid first (SPEP+IFE+FLC) + mass spectrometry fibril typing MANDATORY</li>
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
              <div style={{ fontSize: 11 }}><b>Cardiac:</b> {s.cardiac_n}/40</div>
              <div style={{ fontSize: 11 }}><b>Neuro:</b> {s.neuropathy_n}/40</div>
              <div style={{ fontSize: 11 }}><b>Renal:</b> {s.renal_n}/40</div>
            </div>
          );
        })}
      </div>
      <div style={{ background: '#e8f5e9', border: '2px solid #1b5e20', borderRadius: 8, padding: 16 }}>
        <b>💊 FDA-Approved TTR Therapies:</b>
        <div style={{ marginTop: 8, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {(data.key_clinical_themes?.FDA_approved_TTR_therapies || []).map(t => (
            <Badge key={t} text={t} color='#1b5e20' />
          ))}
        </div>
        <div style={{ marginTop: 12 }}>
          <b>Liver Transplant Curative:</b>{' '}
          {(data.key_clinical_themes?.liver_transplant_curative || []).join(' · ')}
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
            <tr style={{ background: '#1a237e', color: '#fff' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gene</th>
              <th style={{ padding: '8px 12px' }}>Amyloid Type</th>
              <th style={{ padding: '8px 12px' }}>Age Dx</th>
              <th style={{ padding: '8px 12px' }}>Cardiac</th>
              <th style={{ padding: '8px 12px' }}>Neuro</th>
              <th style={{ padding: '8px 12px' }}>Renal</th>
              <th style={{ padding: '8px 12px' }}>Special</th>
              <th style={{ padding: '8px 12px', textAlign: 'left' }}>Key Rule</th>
            </tr>
          </thead>
          <tbody>
            {genes.map((gene, idx) => {
              const d = breakdown.gene_data[gene];
              const color = GENE_COLORS[gene] || '#555';
              const amyloidType = breakdown.amyloid_type_map?.[gene] || '?';
              const curative = breakdown.treatment_curative?.[gene] || '';
              const special = gene === 'GSN' ? `Corneal: ${d.corneal_lattice_n}/40`
                : gene === 'B2M' ? `CTS: ${d.carpal_tunnel_n}/40`
                : gene === 'OSMR' ? `Pruritic: ${d.cutaneous_n}/40`
                : gene === 'LYZ' ? `Hepatomegaly: ${d.hepatomegaly_n}/40`
                : gene === 'FGA' ? 'No neuropathy'
                : '';
              return (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                  <td style={{ padding: '6px 12px' }}>
                    <span style={{ background: color, color: '#fff', padding: '2px 8px', borderRadius: 4, fontWeight: 700 }}>{gene}</span>
                  </td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', fontWeight: 600 }}>{amyloidType}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.mean_age_dx} yr</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.cardiac_n}/40</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.neuropathy_n}/40</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.renal_n}/40</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', fontSize: 11 }}>{special}</td>
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
      <h3>Clinical Atlas — 8 Gene Profiles</h3>
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
      <h3>Definitions & Critical Clinical Rules</h3>
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
        <div key={term} style={{ marginBottom: 12, borderLeft: '3px solid #1a237e', paddingLeft: 12 }}>
          <div style={{ fontWeight: 700, color: '#1a237e' }}>{term}</div>
          <div style={{ fontSize: 13 }}>{defn}</div>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryTransthyretinAmyloidosisAtlasPage() {
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
            background: tab === i ? '#1a237e' : '#e8eaf6', color: tab === i ? '#fff' : '#333',
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
