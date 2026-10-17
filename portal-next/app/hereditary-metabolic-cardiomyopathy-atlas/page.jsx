'use client';
import { useState, useEffect } from 'react';

const API  = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-metabolic-cardiomyopathy-atlas';
const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  'GLA':     '#880e4f',  // deep crimson    -- Fabry; lysosomal; cornea-verticillata; ERT/migalastat
  'LAMP2':   '#1a237e',  // deep navy       -- Danon; TRIAD; massive-HCM; cardiac-transplant
  'GAA':     '#1b5e20',  // dark green      -- Pompe; IOPD-cardiomegaly; ECG-triad; ERT
  'PRKAG2':  '#e65100',  // burnt orange    -- AMPK; WPW+HCM+conduction; ICD-dangerous; ablation-first
  'TAZ':     '#0d47a1',  // medium blue     -- Barth; cardiolipin; DCM+neutropenia+3-MGA
  'SLC22A5': '#4a148c',  // deep purple     -- CDSP; OCTN2; L-carnitine-CURATIVE; NBS
  'HADHA':   '#bf360c',  // dark red-orange -- LCHAD/MTP; AFLP-mother-PATHOGNOMONIC; retinopathy
  'ACAD9':   '#37474f',  // dark blue-grey  -- Complex-I; riboflavin-FIRST-LINE; lactic-acidosis
};

const GENE_INFO = {
  'GLA':     { full: 'Alpha-galactosidase-A / Fabry / HCM-like / cornea-verticillata-PATHOGNOMONIC / ERT-migalastat', locus: 'Xq22.1', size: '429 aa / 46 kDa', inh: 'X-linked' },
  'LAMP2':   { full: 'LAMP2 / Danon / massive-HCM+DCM / TRIAD-PATHOGNOMONIC / cardiac-transplant-only', locus: 'Xq24', size: '410 aa / ~110 kDa glyco', inh: 'X-linked' },
  'GAA':     { full: 'Acid-alpha-glucosidase / Pompe-GSD-II / IOPD-cardiomegaly-PATHOGNOMONIC / ECG-triad / ERT', locus: '17q25.3', size: '952 aa / 110 kDa', inh: 'AR' },
  'PRKAG2':  { full: 'AMPK-gamma-2 / WPW+HCM+conduction-TRIAD-PATHOGNOMONIC / ICD-alone-DANGEROUS-WPW / ablation-FIRST', locus: '7q36.1', size: '569 aa / 63 kDa', inh: 'AD' },
  'TAZ':     { full: 'Tafazzin / Barth / DCM-LVNC+neutropenia+3-MGA-TRIAD-PATHOGNOMONIC / males-only / cardiolipin', locus: 'Xq28', size: '292 aa / 33 kDa', inh: 'X-linked' },
  'SLC22A5': { full: 'OCTN2-carnitine-transporter / Primary-carnitine-deficiency / L-carnitine-CURATIVE / NBS-low-C0', locus: '5q31.1', size: '557 aa / 63 kDa', inh: 'AR' },
  'HADHA':   { full: 'MTP-alpha/LCHAD / DCM+neuropathy+retinopathy / AFLP-heterozygous-mother-PATHOGNOMONIC / MCT-diet', locus: '2p23.3', size: '763 aa / 83 kDa', inh: 'AR' },
  'ACAD9':   { full: 'Complex-I-assembly / riboflavin-responsive-cardiomyopathy / 100-300mg-riboflavin-FIRST-LINE / lactic-acidosis', locus: '3q21.3', size: '621 aa / 68 kDa', inh: 'AR' },
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
        <Badge text={`Myopathy: ${cohort.myopathy_n}/40`} color='#6a1b9a' />
        <Badge text={`Transplant: ${cohort.transplant_n}/40`} color='#37474f' />
      </div>
      {cohort.cornea_verticillata_n > 0 && (
        <div style={{ fontSize: 12, background: '#fce4ec', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          👁 <b>Cornea Verticillata:</b> {cohort.cornea_verticillata_n}/40 — slit-lamp whorl opacities PATHOGNOMONIC Fabry
        </div>
      )}
      {cohort.wpw_n > 0 && (
        <div style={{ fontSize: 12, background: '#fff3e0', padding: '4px 8px', borderRadius: 4, marginBottom: 6, fontWeight: 700, color: '#e65100' }}>
          ⚡ <b>WPW:</b> {cohort.wpw_n}/40 — CATHETER ABLATION FIRST before ICD (ICD alone dangerous)
        </div>
      )}
      {cohort.neutropenia_n > 0 && (
        <div style={{ fontSize: 12, background: '#e3f2fd', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🩸 <b>Neutropenia:</b> {cohort.neutropenia_n}/40 — cyclic/persistent + 3-MGA urine = Barth PATHOGNOMONIC
        </div>
      )}
      {cohort.retinopathy_n > 0 && (
        <div style={{ fontSize: 12, background: '#e8f5e9', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          👁 <b>Retinopathy:</b> {cohort.retinopathy_n}/40 — pigmentary chorioretinal (LCHAD) — annual ophthalmology
        </div>
      )}
      {cohort.hypoketotic_hypoglycemia_n > 0 && (
        <div style={{ fontSize: 12, background: '#fffde7', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🍬 <b>Hypoketotic Hypoglycemia:</b> {cohort.hypoketotic_hypoglycemia_n}/40 — low glucose + absent ketones PATHOGNOMONIC
        </div>
      )}
      {cohort.lactic_acidosis_n > 0 && (
        <div style={{ fontSize: 12, background: '#f3e5f5', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🧪 <b>Lactic Acidosis:</b> {cohort.lactic_acidosis_n}/40 — Complex I → riboflavin 100-300mg/day FIRST
        </div>
      )}
      {cohort.intellectual_disability_n > 0 && (
        <div style={{ fontSize: 12, background: '#e8eaf6', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          🧠 <b>Intellectual Disability:</b> {cohort.intellectual_disability_n}/40 — Danon triad component (males ~70%)
        </div>
      )}
      {cohort.respiratory_n > 0 && (
        <div style={{ fontSize: 12, background: '#e0f7fa', padding: '4px 8px', borderRadius: 4, marginBottom: 6 }}>
          💨 <b>Respiratory:</b> {cohort.respiratory_n}/40 — Pompe LOPD: diaphragm failure primary mortality
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
      <div style={{ background: '#880e4f', color: '#fff', borderRadius: 8, padding: 20, marginBottom: 20 }}>
        <h2 style={{ margin: 0, fontSize: 22 }}>🧬 Hereditary Metabolic Cardiomyopathy Atlas</h2>
        <p style={{ margin: '8px 0 0', opacity: 0.9, fontSize: 13 }}>{data.subtitle}</p>
        <div style={{ marginTop: 12, display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          <Badge text={`${data.total_patients} Patients`} color='#6d0033' />
          <Badge text={`8 Genes`} color='#4a148c' />
          <Badge text={`Seeds ${data.seeds}`} color='#1b5e20' />
        </div>
      </div>
      <div style={{ background: '#fff3e0', border: '2px solid #e65100', borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <b>🚨 Key Clinical Rules:</b>
        <ul style={{ margin: '8px 0 0', paddingLeft: 20, fontSize: 13 }}>
          <li><b>GLA/Fabry:</b> CMR T1 LOW + basal-inferolateral LGE = PATHOGNOMONIC — enzyme assay normal in females → gene sequencing MANDATORY</li>
          <li><b>LAMP2/Danon:</b> Massive HCM + myopathy + intellectual disability TRIAD — cardiac transplant ONLY definitive therapy</li>
          <li><b>GAA/Pompe IOPD:</b> Short PR + high voltage + left axis = ECG TRIAD PATHOGNOMONIC — CRIM status before ERT</li>
          <li><b>PRKAG2:</b> WPW + HCM + conduction disease TRIAD — ICD alone DANGEROUS (anti-tachycardia pacing → AF → VF) — catheter ablation FIRST</li>
          <li><b>TAZ/Barth:</b> DCM/LVNC + neutropenia + 3-methylglutaconic aciduria = TRIAD PATHOGNOMONIC — males only</li>
          <li><b>SLC22A5:</b> NBS low C0 (&lt;5 μM) — L-carnitine CURATIVE — avoid pivalate antibiotics</li>
          <li><b>HADHA/LCHAD:</b> Mother with AFLP/HELLP → TEST BABY FOR LCHAD IMMEDIATELY (PATHOGNOMONIC association)</li>
          <li><b>ACAD9:</b> Riboflavin 100-300mg/day FIRST-LINE ALWAYS before all other interventions — avoid metformin</li>
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
              <div style={{ fontSize: 11 }}><b>Myopathy:</b> {s.myopathy_n}/40</div>
              <div style={{ fontSize: 11 }}><b>Transplant:</b> {s.transplant_n}/40</div>
            </div>
          );
        })}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={{ background: '#e8f5e9', border: '2px solid #1b5e20', borderRadius: 8, padding: 16 }}>
          <b>💊 ERT Available:</b>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.enzyme_replacement_available || []).map(t => (
              <Badge key={t} text={t} color='#1b5e20' />
            ))}
          </div>
          <div style={{ marginTop: 10 }}><b>💊 Small Molecule:</b></div>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.small_molecule_therapy || []).map(t => (
              <Badge key={t} text={t} color='#0d47a1' />
            ))}
          </div>
        </div>
        <div style={{ background: '#fce4ec', border: '2px solid #880e4f', borderRadius: 8, padding: 16 }}>
          <b>⚡ WPW-associated:</b>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.wpw_present || []).map(t => (
              <Badge key={t} text={t} color='#e65100' />
            ))}
          </div>
          <div style={{ marginTop: 10 }}><b>🏥 Transplant often needed:</b></div>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.cardiac_transplant_often_needed || []).map(t => (
              <Badge key={t} text={t} color='#37474f' />
            ))}
          </div>
          <div style={{ marginTop: 10 }}><b>🩺 NBS detected:</b></div>
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {(data.key_clinical_themes?.NBS_detected || []).map(t => (
              <Badge key={t} text={t} color='#0277bd' />
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
            <tr style={{ background: '#880e4f', color: '#fff' }}>
              <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gene</th>
              <th style={{ padding: '8px 12px' }}>Disease</th>
              <th style={{ padding: '8px 12px' }}>Inh</th>
              <th style={{ padding: '8px 12px' }}>Age Dx</th>
              <th style={{ padding: '8px 12px' }}>Cardiac</th>
              <th style={{ padding: '8px 12px' }}>Myopathy</th>
              <th style={{ padding: '8px 12px' }}>Transplant</th>
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
                GLA: 'Fabry', LAMP2: 'Danon', GAA: 'Pompe/GSD-II',
                PRKAG2: 'PRKAG2-CMP', TAZ: 'Barth', SLC22A5: 'CDSP',
                HADHA: 'LCHAD/MTP', ACAD9: 'Complex-I-CMP',
              };
              return (
                <tr key={gene} style={{ background: idx % 2 === 0 ? '#f5f5f5' : '#fff' }}>
                  <td style={{ padding: '6px 12px' }}>
                    <span style={{ background: color, color: '#fff', padding: '2px 8px', borderRadius: 4, fontWeight: 700 }}>{gene}</span>
                  </td>
                  <td style={{ padding: '6px 12px', fontWeight: 600 }}>{diseaseMap[gene]}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center', fontSize: 11 }}>{inh}</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.mean_age_dx} yr</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.cardiac_n}/40</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.myopathy_n}/40</td>
                  <td style={{ padding: '6px 12px', textAlign: 'center' }}>{d.transplant_n}/40</td>
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
        <div key={term} style={{ marginBottom: 12, borderLeft: '3px solid #880e4f', paddingLeft: 12 }}>
          <div style={{ fontWeight: 700, color: '#880e4f' }}>{term}</div>
          <div style={{ fontSize: 13 }}>{defn}</div>
        </div>
      ))}
    </div>
  );
}

export default function HereditaryMetabolicCardiomyopathyAtlasPage() {
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
            background: tab === i ? '#880e4f' : '#fce4ec', color: tab === i ? '#fff' : '#333',
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
