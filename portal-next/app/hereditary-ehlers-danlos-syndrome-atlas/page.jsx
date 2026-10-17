'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-ehlers-danlos-syndrome-atlas';
const TABS = ['Overview', 'Gene Table', 'EDS Classification', 'Definitions'];

const GENE_COLORS = {
  'COL5A1':   '#1565c0',  // deep blue      -- cEDS most common; atrophic scar PATHOGNOMONIC
  'COL5A2':   '#1976d2',  // medium blue    -- cEDS type 2; allelic COL5A1
  'TNXB':     '#6a1b9a',  // deep purple    -- TNX-EDS hEDS-like; 5-10% hEDS have TNXB
  'PLOD1':    '#b71c1c',  // dark red       -- kEDS-PLOD1; LP/HLP ratio PATHOGNOMONIC; vascular risk
  'FKBP14':   '#880e4f',  // deep crimson   -- kEDS-FKBP14; hearing loss PATHOGNOMONIC
  'ADAMTS2':  '#e65100',  // burnt orange   -- dEDS; extreme skin fragility PATHOGNOMONIC
  'CHST14':   '#1b5e20',  // dark green     -- mcEDS; adducted thumbs at birth PATHOGNOMONIC
  'SLC39A13': '#37474f',  // blue-grey      -- spEDS; fish-eye cornea PATHOGNOMONIC
};

const GENE_INFO = {
  'COL5A1':   { full: 'Collagen-type-V-alpha1 / cEDS-most-common-90pct / AD-haploinsufficiency / atrophic-papyraceous-scar-PATHOGNOMONIC / skin-hyperextensibility / NO-vascular-rupture', locus: '9q34.3', size: '1838 aa / 184 kDa', inh: 'AD' },
  'COL5A2':   { full: 'Collagen-type-V-alpha2 / cEDS-type2 / AD-LOF-glycine-substitutions / same-phenotype-COL5A1 / rarer-5pct-cEDS', locus: '2q31.1', size: '1499 aa / 145 kDa', inh: 'AD' },
  'TNXB':     { full: 'Tenascin-XB / TNX-EDS-hEDS-like / AR-haploinsufficiency / minimal-scarring / TNXB-panel-mandatory-hEDS / biallelic-Clarkson-capillary-leak', locus: '6p21.32', size: '4243 aa / 450 kDa', inh: 'AR' },
  'PLOD1':    { full: 'Lysyl-hydroxylase-1 / kEDS-PLOD1-most-common / urine-LP-HLP-ratio-gt3x-PATHOGNOMONIC / VASCULAR-RUPTURE-RISK / neonatal-hypotonia-kyphoscoliosis / aortic-imaging-MANDATORY', locus: '1p36.22', size: '727 aa / 83 kDa', inh: 'AR' },
  'FKBP14':   { full: 'FKBP-prolyl-isomerase-14 / kEDS-FKBP14 / hearing-loss-PATHOGNOMONIC / urine-LP-ratio-NORMAL / no-vascular-rupture / kyphoscoliosis-hearing-loss', locus: '7p14.3', size: '174 aa / 16 kDa', inh: 'AR' },
  'ADAMTS2':  { full: 'Procollagen-N-proteinase / dEDS-dermatosparaxis / extreme-skin-fragility-PATHOGNOMONIC / sagging-redundant-skin-birth / skin-NOT-hyperextensible / PROM', locus: '5q35.3', size: '1211 aa / 138 kDa', inh: 'AR' },
  'CHST14':   { full: 'Dermatan-sulfate-sulfotransferase / mcEDS / adducted-thumbs-clubfoot-birth-PATHOGNOMONIC / progressive-joint-contractures / glycosaminoglycan-defect', locus: '15q15.1', size: '376 aa / 43 kDa', inh: 'AR' },
  'SLC39A13': { full: 'ZIP13-zinc-transporter / spEDS / protruding-eyes-fish-eye-cornea-PATHOGNOMONIC / short-stature-skeletal-dysplasia / zinc-homeostasis-ECM', locus: '11p11.2', size: '356 aa / 40 kDa', inh: 'AR' },
};

const EDS_CLASSIFICATION = [
  { gene: 'COL5A1', eds_type: 'cEDS', subtype: 'Classical EDS', key_feature: 'Atrophic papyraceous scars', vascular_risk: 'LOW', urine_test: 'N/A', hearing_loss: 'No', onset: 'Childhood' },
  { gene: 'COL5A2', eds_type: 'cEDS', subtype: 'Classical EDS type 2', key_feature: 'Same as COL5A1 (rarer)', vascular_risk: 'LOW', urine_test: 'N/A', hearing_loss: 'No', onset: 'Childhood' },
  { gene: 'TNXB',   eds_type: 'TNX-EDS', subtype: 'Tenascin-X deficient', key_feature: 'hEDS phenotype; minimal scars', vascular_risk: 'LOW', urine_test: 'N/A', hearing_loss: 'No', onset: 'Adult' },
  { gene: 'PLOD1',  eds_type: 'kEDS-1', subtype: 'Kyphoscoliotic type 1', key_feature: 'LP/HLP ratio >3x; vascular risk', vascular_risk: 'HIGH', urine_test: 'LP/HLP >3x PATHOGNOMONIC', hearing_loss: 'No', onset: 'Neonatal' },
  { gene: 'FKBP14', eds_type: 'kEDS-2', subtype: 'Kyphoscoliotic type 2', key_feature: 'Sensorineural hearing loss', vascular_risk: 'LOW', urine_test: 'LP/HLP NORMAL', hearing_loss: 'YES — PATHOGNOMONIC', onset: 'Neonatal' },
  { gene: 'ADAMTS2',eds_type: 'dEDS', subtype: 'Dermatosparaxis EDS', key_feature: 'Extreme skin fragility; NOT hyperextensible', vascular_risk: 'LOW', urine_test: 'N/A', hearing_loss: 'No', onset: 'Birth' },
  { gene: 'CHST14', eds_type: 'mcEDS', subtype: 'Musculocontractural EDS', key_feature: 'Adducted thumbs + clubfoot at birth', vascular_risk: 'LOW', urine_test: 'Urine GAG analysis', hearing_loss: 'No', onset: 'Birth' },
  { gene: 'SLC39A13',eds_type: 'spEDS', subtype: 'Spondylodysplastic EDS', key_feature: 'Fish-eye cornea; protruding eyes', vascular_risk: 'LOW', urine_test: 'N/A', hearing_loss: 'No', onset: 'Birth' },
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
          <div style={{ fontWeight: 600 }}>{cohort.eds_type || ''}</div>
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
          {cohort.skin_hyperextensibility_n > 0 && <Badge text={`skin-ext ${cohort.skin_hyperextensibility_n}/40`} color="#1565c0" />}
          {cohort.atrophic_scar_n > 0 && <Badge text={`scar ${cohort.atrophic_scar_n}/40`} color="#6a1b9a" />}
          {cohort.joint_hypermobility_n > 0 && <Badge text={`joint-lax ${cohort.joint_hypermobility_n}/40`} color="#2e7d32" />}
          {cohort.kyphoscoliosis_n > 0 && <Badge text={`kypho ${cohort.kyphoscoliosis_n}/40`} color="#b71c1c" />}
          {cohort.hearing_loss_n > 0 && <Badge text={`hearing-loss ${cohort.hearing_loss_n}/40`} color="#880e4f" />}
          {cohort.adducted_thumbs_n > 0 && <Badge text={`adducted-thumbs ${cohort.adducted_thumbs_n}/40`} color="#1b5e20" />}
          {cohort.protruding_eyes_n > 0 && <Badge text={`protruding-eyes ${cohort.protruding_eyes_n}/40`} color="#37474f" />}
          {cohort.vascular_rupture_n > 0 && <Badge text={`vasc-rupture ${cohort.vascular_rupture_n}/40`} color="#c62828" />}
        </div>
      )}
    </div>
  );
}

export default function HEDSAtlasPage() {
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

  if (loading) return <div style={{ padding: 40, color: '#666' }}>Loading Hereditary Ehlers-Danlos Syndrome Atlas…</div>;
  if (error)   return <div style={{ padding: 40, color: 'red' }}>Error: {error}</div>;

  const genes = overview?.genes || [];

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1100, margin: '0 auto', padding: 20 }}>
      {/* Header */}
      <div style={{ background: 'linear-gradient(135deg,#1565c0,#4a148c)', color: '#fff', borderRadius: 10, padding: 24, marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22 }}>🧬 Hereditary Ehlers-Danlos Syndrome Atlas</h1>
        <p style={{ margin: '8px 0 0', opacity: 0.9, fontSize: 13 }}>
          Complete 8-Gene Reference · COL5A1/cEDS · COL5A2 · TNXB/TNX-EDS · PLOD1/kEDS · FKBP14/kEDS · ADAMTS2/dEDS · CHST14/mcEDS · SLC39A13/spEDS
        </p>
        <p style={{ margin: '4px 0 0', opacity: 0.75, fontSize: 12 }}>
          320 patients · 8×40 cohorts · seeds {overview?.seeds}
        </p>
      </div>

      {/* Critical Rules Banner */}
      <div style={{ background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 8, padding: 12, marginBottom: 20 }}>
        <strong style={{ color: '#b8860b' }}>⚠️ Critical Clinical Rules</strong>
        <ul style={{ margin: '6px 0 0', paddingLeft: 20, fontSize: 12 }}>
          <li><strong>COL5A1-CEDS:</strong> Atrophic papyraceous scars = PATHOGNOMONIC. NO vascular rupture risk (distinguish from COL3A1/vEDS).</li>
          <li><strong>PLOD1-kEDS:</strong> Urine LP/HLP ratio &gt;3× = PATHOGNOMONIC. Vascular rupture risk — aortic imaging MANDATORY annually.</li>
          <li><strong>FKBP14-kEDS:</strong> Sensorineural hearing loss = PATHOGNOMONIC vs PLOD1. LP/HLP urine ratio NORMAL. No vascular risk.</li>
          <li><strong>ADAMTS2-dEDS:</strong> Extreme skin fragility + sagging skin at birth = PATHOGNOMONIC. Skin NOT hyperextensible (unlike cEDS).</li>
          <li><strong>CHST14-mcEDS:</strong> Adducted thumbs + clubfoot at birth = PATHOGNOMONIC. Order CHST14 immediately in neonatal contractures.</li>
          <li><strong>TNXB:</strong> Panel MANDATORY in hEDS workup — 5–10% of hEDS patients have detectable TNXB variant.</li>
        </ul>
      </div>

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            style={{ padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: activeTab === t ? '#1565c0' : '#eee', color: activeTab === t ? '#fff' : '#333', fontWeight: activeTab === t ? 700 : 400 }}>
            {t}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'Overview' && (
        <div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 20 }}>
            {[
              { label: 'Total Patients', value: overview?.total_patients },
              { label: 'EDS Subtypes', value: genes.length },
              { label: 'Cohort Size', value: overview?.cohort_size + ' / gene' },
              { label: 'Seeds', value: overview?.seeds },
            ].map(({ label, value }) => (
              <div key={label} style={{ background: '#f5f5f5', borderRadius: 8, padding: 16, textAlign: 'center' }}>
                <div style={{ fontSize: 24, fontWeight: 700, color: '#1565c0' }}>{value}</div>
                <div style={{ fontSize: 12, color: '#666' }}>{label}</div>
              </div>
            ))}
          </div>

          <h3>Key Facts</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 20 }}>
            {Object.entries(overview?.key_facts || {}).map(([k, v]) => (
              <div key={k} style={{ background: '#f9f9f9', border: '1px solid #ddd', borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 600, fontSize: 12, color: '#444', marginBottom: 4 }}>{k.replace(/_/g,' ').toUpperCase()}</div>
                <div style={{ fontSize: 13 }}>{v}</div>
              </div>
            ))}
          </div>

          <h3>Diagnostic Algorithm</h3>
          <div style={{ background: '#e3f2fd', border: '1px solid #90caf9', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            {Object.entries(overview?.diagnostic_approach || {}).map(([k, v]) => (
              <div key={k} style={{ marginBottom: 8 }}>
                <span style={{ fontWeight: 600, fontSize: 12 }}>{k.replace(/_/g,' ')}: </span>
                <span style={{ fontSize: 12 }}>{v}</span>
              </div>
            ))}
          </div>

          <h3>Gene Cards</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            {genes.map(g => (
              <GeneCard key={g} gene={g} color={GENE_COLORS[g] || '#555'} info={GENE_INFO[g]} breakdown={breakdown} />
            ))}
          </div>
        </div>
      )}

      {/* Gene Table Tab */}
      {activeTab === 'Gene Table' && (
        <div>
          <h3>8-Gene Reference Table</h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#1565c0', color: '#fff' }}>
                  {['Gene','EDS Type','Locus','Size','Inh','Onset (yr)','Dx Lag (yr)','Skin Ext n/40','Scar n/40','Joint Lax n/40','Kypho n/40','Hearing n/40','Adducted n/40','Cornea n/40','Vasc Rupture n/40'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {genes.map((g, idx) => {
                  const d = breakdown?.gene_data?.[g] || {};
                  return (
                    <tr key={g} style={{ background: idx % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                      <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[g] || '#333' }}>{g}</td>
                      <td style={{ padding: '6px 10px' }}>{d.eds_type}</td>
                      <td style={{ padding: '6px 10px' }}>{d.locus}</td>
                      <td style={{ padding: '6px 10px' }}>{GENE_INFO[g]?.size}</td>
                      <td style={{ padding: '6px 10px' }}>{d.inheritance}</td>
                      <td style={{ padding: '6px 10px' }}>{d.mean_onset_age}</td>
                      <td style={{ padding: '6px 10px' }}>{d.mean_dx_lag_yrs}</td>
                      <td style={{ padding: '6px 10px' }}>{d.skin_hyperextensibility_n}</td>
                      <td style={{ padding: '6px 10px' }}>{d.atrophic_scar_n}</td>
                      <td style={{ padding: '6px 10px' }}>{d.joint_hypermobility_n}</td>
                      <td style={{ padding: '6px 10px' }}>{d.kyphoscoliosis_n}</td>
                      <td style={{ padding: '6px 10px' }}>{d.hearing_loss_n}</td>
                      <td style={{ padding: '6px 10px' }}>{d.adducted_thumbs_n}</td>
                      <td style={{ padding: '6px 10px' }}>{d.corneal_abnormality_n}</td>
                      <td style={{ padding: '6px 10px', fontWeight: d.vascular_rupture_n > 0 ? 700 : 400, color: d.vascular_rupture_n > 0 ? 'red' : 'inherit' }}>{d.vascular_rupture_n}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* EDS Classification Tab */}
      {activeTab === 'EDS Classification' && (
        <div>
          <h3>EDS Subtype Classification & Diagnostic Distinctions</h3>
          <div style={{ overflowX: 'auto', marginBottom: 24 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: '#4a148c', color: '#fff' }}>
                  {['Gene','EDS Type','Subtype','Key Pathognomonic Feature','Vascular Risk','Urine Test','Hearing Loss','Onset'].map(h => (
                    <th key={h} style={{ padding: '8px 10px', textAlign: 'left', whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {EDS_CLASSIFICATION.map((row, idx) => (
                  <tr key={row.gene} style={{ background: idx % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                    <td style={{ padding: '6px 10px', fontWeight: 700, color: GENE_COLORS[row.gene] || '#333' }}>{row.gene}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 600 }}>{row.eds_type}</td>
                    <td style={{ padding: '6px 10px' }}>{row.subtype}</td>
                    <td style={{ padding: '6px 10px', fontStyle: 'italic' }}>{row.key_feature}</td>
                    <td style={{ padding: '6px 10px', color: row.vascular_risk === 'HIGH' ? 'red' : 'green', fontWeight: row.vascular_risk === 'HIGH' ? 700 : 400 }}>{row.vascular_risk}</td>
                    <td style={{ padding: '6px 10px' }}>{row.urine_test}</td>
                    <td style={{ padding: '6px 10px', color: row.hearing_loss.startsWith('YES') ? '#880e4f' : 'inherit', fontWeight: row.hearing_loss.startsWith('YES') ? 700 : 400 }}>{row.hearing_loss}</td>
                    <td style={{ padding: '6px 10px' }}>{row.onset}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h3>Classification by EDS Group</h3>
          {Object.entries(breakdown?.eds_classification || {}).map(([type, data]) => (
            <div key={type} style={{ border: '1px solid #ddd', borderRadius: 8, padding: 14, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6, color: '#1565c0' }}>{type.replace(/_/g, ' ').toUpperCase()}</div>
              <div style={{ fontSize: 12 }}><strong>Genes:</strong> {data.genes?.join(', ')}</div>
              <div style={{ fontSize: 12 }}><strong>Hallmarks:</strong> {data.hallmarks}</div>
              {data.distinguish && <div style={{ fontSize: 12, color: '#880e4f' }}><strong>Distinguish:</strong> {data.distinguish}</div>}
              {data.vascular_risk && <div style={{ fontSize: 12, color: data.vascular_risk === 'HIGH for PLOD1 (like vEDS); LOW for FKBP14' ? '#c62828' : '#2e7d32' }}><strong>Vascular Risk:</strong> {data.vascular_risk}</div>}
              {data.test_in && <div style={{ fontSize: 12, color: '#6a1b9a' }}><strong>Test in:</strong> {data.test_in}</div>}
            </div>
          ))}
        </div>
      )}

      {/* Definitions Tab */}
      {activeTab === 'Definitions' && (
        <div>
          <h3>Clinical Definitions & Key Concepts</h3>
          {Object.entries(defs?.definitions || {}).map(([term, defn]) => (
            <div key={term} style={{ border: '1px solid #e0e0e0', borderRadius: 8, padding: 14, marginBottom: 12 }}>
              <div style={{ fontWeight: 700, fontSize: 14, color: '#1565c0', marginBottom: 6 }}>{term}</div>
              <div style={{ fontSize: 13, lineHeight: 1.6 }}>{defn}</div>
            </div>
          ))}

          <h3 style={{ marginTop: 24 }}>Critical Clinical Rules ({Object.keys(defs?.critical_clinical_rules || {}).length} rules)</h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {Object.keys(defs?.critical_clinical_rules || {}).map(rule => (
              <span key={rule} style={{ background: '#e8eaf6', border: '1px solid #9fa8da', borderRadius: 4, padding: '3px 8px', fontSize: 11 }}>
                {rule.replace(/-/g,' ')}
              </span>
            ))}
          </div>

          <h3 style={{ marginTop: 24 }}>Cascade Testing Sequences</h3>
          {Object.entries(defs?.cascade_testing || {}).map(([gene, steps]) => (
            <div key={gene} style={{ marginBottom: 10 }}>
              <span style={{ fontWeight: 700, color: GENE_COLORS[gene] || '#333', marginRight: 8 }}>{gene}</span>
              {steps.map((s, i) => (
                <span key={i} style={{ background: '#f5f5f5', borderRadius: 4, padding: '2px 7px', fontSize: 11, marginRight: 4 }}>{s.replace(/_/g,' ')}</span>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
