'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-skeletal-dysplasia-atlas';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FGFR3:   '#1565c0',  // deep blue    — ACH G380R foramen magnum vosoritide
  COL2A1:  '#2e7d32',  // deep green   — Stickler membranous vitreous retinal detachment
  EXT1:    '#b71c1c',  // deep red     — MHE1 chondrosarcoma highest risk
  EXT2:    '#880e4f',  // deep magenta — MHE2 milder same surveillance
  COMP:    '#e65100',  // deep orange  — PSACH normal face CRITICAL DDx ACH
  SLC26A2: '#6a1b9a',  // deep purple  — DTD cauliflower ear hitchhiker thumb
  TRPV4:   '#004d40',  // deep teal    — metatropic dumbbell metaphyses kyphoscoliosis
  ACAN:    '#37474f',  // dark slate   — FSS advanced bone age normal GH
};

const GENE_INFO = {
  FGFR3:   { aa: 806,  locus: '4p16.3',   inh: 'AD',    disease: 'Achondroplasia-G380R-De-Novo-98pct — Foramen-Magnum-Stenosis-Screen-Year-1 — Vosoritide-FDA-2021 — L1→L5-Interpediculate-Narrowing-PATHOGNOMONIC — TD-Telephone-Receiver-Femur-Lethal' },
  COL2A1:  { aa: 1487, locus: '12q13.11', inh: 'AD',    disease: 'Stickler-Type-1-Membranous-Vitreous-PATHOGNOMONIC — Retinal-Detachment-Laser-MANDATORY — SEDC-Odontoid-Hypoplasia-C-SPINE-CI — Pierre-Robin-Neonatal-Airway — Early-OA-30s-40s' },
  EXT1:    { aa: 746,  locus: '8q24.11',  inh: 'AD',    disease: 'MHE1-Osteochondroma — Chondrosarcoma-1-2pct-HIGHEST-RISK — Cap->2cm-BIOPSY-URGENTLY — Rapid-Growth-Post-Skeletal-Maturity-Sarcoma — Two-Hit-Tumour-Suppressor' },
  EXT2:    { aa: 718,  locus: '11p12-p11',inh: 'AD',    disease: 'MHE2-Milder-Than-EXT1 — Same-Surveillance-Protocol — Malignant-Transformation-0.5pct — EXT1-EXT2-Heterodimer-HSPG — Never-Reassure-Zero-Sarcoma-Risk' },
  COMP:    { aa: 757,  locus: '19p13.11', inh: 'AD',    disease: 'PSACH-NORMAL-Face-NORMAL-Head-CRITICAL-DDx-ACH — Not-Apparent-At-Birth-Ambulation-18M — C1-C2-Instability-MANDATORY — Contact-Sports-ABSOLUTELY-CI — MED-Early-OA-Childhood' },
  SLC26A2: { aa: 739,  locus: '5q32',     inh: 'AR',    disease: 'DTD-Cauliflower-Ear-PATHOGNOMONIC — Hitchhiker-Thumb-Bilateral-PATHOGNOMONIC — Bilateral-Club-Foot-Invariant — Cervical-Kyphosis-Cord-Compression-Neonatal — ACG1B-Lethal-Null' },
  TRPV4:   { aa: 871,  locus: '12q24.11', inh: 'AD',    disease: 'Metatropic-Dumbbell-Metaphyses-PATHOGNOMONIC — Changing-Proportions-Metatropic-Name — Platyspondyly-All-Spectrum — Respiratory-Failure-Main-Mortality — GOF-Brachyolmia-Mild-End' },
  ACAN:    { aa: 2153, locus: '15q26.1',  inh: 'AD/AR', disease: 'FSS-Advanced-Bone-Age-PATHOGNOMONIC — Normal-GH-Axis-DISTINGUISH-GH-Deficiency — GH-Therapy-Limited-By-Bone-Age — OCD-Osteochondritis-Dissecans — SED-Kimberley-AR-SNHL' },
};

function Loading() {
  return <div style={{ padding: 20, color: '#555' }}>Loading…</div>;
}
function ErrBox({ msg }) {
  return <div style={{ padding: 16, background: '#fce4ec', color: '#b71c1c', borderRadius: 6 }}>{msg}</div>;
}
function StatCard({ label, value, color }) {
  return (
    <div style={{ background: '#fff', border: `2px solid ${color || '#1565c0'}`, borderRadius: 8, padding: '12px 16px', minWidth: 160, textAlign: 'center' }}>
      <div style={{ fontSize: 26, fontWeight: 700, color: color || '#1565c0' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#555', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const ov = data;
  return (
    <div>
      <h2 style={{ color: '#1565c0' }}>Hereditary-Skeletal-Dysplasia-Atlas</h2>
      <p style={{ color: '#444', marginBottom: 16 }}>{ov.subtitle}</p>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
        <StatCard label="Total Patients" value={ov.total_patients} color="#1565c0" />
        <StatCard label="Genes" value={ov.genes?.length} color="#2e7d32" />
        <StatCard label="Seeds" value={ov.seeds} color="#37474f" />
        <StatCard label="Short Stature" value={ov.short_stature_patients} color="#1565c0" />
        <StatCard label="Early OA" value={ov.early_oa_patients} color="#e65100" />
        <StatCard label="Scoliosis" value={ov.scoliosis_patients} color="#6a1b9a" />
        <StatCard label="Osteochondroma" value={ov.osteochondroma_patients} color="#b71c1c" />
        <StatCard label="Retinal Detachment" value={ov.retinal_detachment_patients} color="#2e7d32" />
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 24 }}>
        <StatCard label="Foramen Magnum Stenosis" value={ov.foramen_magnum_stenosis_patients} color="#b71c1c" />
        <StatCard label="Sleep Apnoea" value={ov.sleep_apnoea_patients} color="#e65100" />
        <StatCard label="Spinal Stenosis" value={ov.spinal_stenosis_patients} color="#004d40" />
        <StatCard label="Joint Laxity" value={ov.joint_laxity_patients} color="#880e4f" />
        <StatCard label="Cauliflower Ear" value={ov.cauliflower_ear_patients} color="#6a1b9a" />
        <StatCard label="Advanced Bone Age" value={ov.advanced_bone_age_patients} color="#37474f" />
        <StatCard label="Malignant Transform." value={ov.malignant_transformation_patients} color="#b71c1c" />
        <StatCard label="Club Foot" value={ov.club_foot_patients} color="#004d40" />
      </div>

      <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 16, marginBottom: 16 }}>
        <strong>Pathway:</strong> {ov.pathway}
      </div>
      <div style={{ background: '#fce4ec', borderRadius: 8, padding: 16 }}>
        <strong>Key Clinical Insight:</strong> {ov.key_clinical_insight}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  if (!data) return <Loading />;
  const genes = Object.keys(data);
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr style={{ background: '#1565c0', color: '#fff' }}>
            <th style={{ padding: '8px 12px', textAlign: 'left' }}>Gene</th>
            <th>Locus</th>
            <th>Size</th>
            <th>Inh.</th>
            <th>Short St.</th>
            <th>Macroceph.</th>
            <th>Foram. Mg</th>
            <th>OSA</th>
            <th>Sp. Sten.</th>
            <th>Scoliosis</th>
            <th>Early OA</th>
            <th>Ret. Det.</th>
            <th>Osteoch.</th>
            <th>Laxity</th>
            <th>Cauli. Ear</th>
            <th>Adv. BA</th>
            <th>Malig.</th>
            <th>Club Ft.</th>
            <th>Cleft Pal.</th>
            <th>N</th>
          </tr>
        </thead>
        <tbody>
          {genes.map((g, i) => {
            const d = data[g];
            const color = GENE_COLORS[g] || '#333';
            const bg = i % 2 === 0 ? '#fff' : '#f5f5f5';
            const pct = v => `${v}%`;
            return (
              <tr key={g} style={{ background: bg }}>
                <td style={{ padding: '6px 12px', fontWeight: 700, color }}>{g}</td>
                <td style={{ textAlign: 'center' }}>{d.locus}</td>
                <td style={{ textAlign: 'center' }}>{d.protein_size}</td>
                <td style={{ textAlign: 'center' }}>{d.inheritance?.split(' ')[0]}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.rhizomelic_short_stature_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.macrocephaly_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.foramen_magnum_stenosis_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.sleep_apnoea_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.spinal_stenosis_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.scoliosis_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.early_oa_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.retinal_detachment_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.osteochondroma_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.joint_laxity_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.cauliflower_ear_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.advanced_bone_age_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.malignant_transformation_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.club_foot_pct)}</td>
                <td style={{ textAlign: 'center' }}>{pct(d.cleft_palate_pct)}</td>
                <td style={{ textAlign: 'center', fontWeight: 600 }}>{d.n_patients}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;
  const [sel, setSel] = useState(Object.keys(data)[0]);
  const d = data[sel];
  const color = GENE_COLORS[sel] || '#333';
  return (
    <div style={{ display: 'flex', gap: 16 }}>
      <div style={{ minWidth: 110 }}>
        {Object.keys(data).map(g => (
          <div
            key={g}
            onClick={() => setSel(g)}
            style={{
              padding: '8px 14px', marginBottom: 4, borderRadius: 6, cursor: 'pointer',
              background: sel === g ? GENE_COLORS[g] : '#f5f5f5',
              color: sel === g ? '#fff' : '#333',
              fontWeight: sel === g ? 700 : 400,
            }}
          >{g}</div>
        ))}
      </div>
      <div style={{ flex: 1 }}>
        <h3 style={{ color }}>{d.gene} — {d.protein_size} — {d.locus} — {d.inheritance?.split(' ')[0]}</h3>
        <div style={{ marginBottom: 12 }}>
          <strong style={{ color }}>Disease: </strong>
          <span>{GENE_INFO[sel]?.disease}</span>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 16 }}>
          {d.critical_flags?.map(f => (
            <span key={f} style={{ background: color, color: '#fff', borderRadius: 4, padding: '2px 8px', fontSize: 11 }}>{f}</span>
          ))}
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Age / Onset:</strong>
          <p style={{ marginTop: 4, color: '#333' }}>{d.age_of_onset}</p>
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Key Biomarker:</strong>
          <p style={{ marginTop: 4, color: '#333' }}>{d.key_biomarker}</p>
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Pathognomonic / DDx:</strong>
          <p style={{ marginTop: 4, color: '#333' }}>{d.pathognomonic}</p>
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Treatment:</strong>
          <p style={{ marginTop: 4, color: '#333' }}>{d.treatment}</p>
        </div>
        <div style={{ marginBottom: 10 }}>
          <strong>Patient Preview (first 5 of 40):</strong>
          <div style={{ overflowX: 'auto', marginTop: 8 }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 12, width: '100%' }}>
              <thead>
                <tr style={{ background: color, color: '#fff' }}>
                  <th style={{ padding: '4px 8px' }}>ID</th>
                  <th>Age</th>
                  <th>Sex</th>
                  <th>Short St.</th>
                  <th>Scoliosis</th>
                  <th>Early OA</th>
                  <th>Osteoch.</th>
                  <th>Cauli. Ear</th>
                  <th>Adv. BA</th>
                </tr>
              </thead>
              <tbody>
                {d.cohort_preview?.map((p, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#fff' : '#f5f5f5' }}>
                    <td style={{ padding: '3px 8px' }}>{p.patient_id}</td>
                    <td style={{ textAlign: 'center' }}>{p.age}</td>
                    <td style={{ textAlign: 'center' }}>{p.sex}</td>
                    <td style={{ textAlign: 'center' }}>{p.rhizomelic_short_stature ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.scoliosis ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.early_oa ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.osteochondroma ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.cauliflower_ear ? '✓' : ''}</td>
                    <td style={{ textAlign: 'center' }}>{p.advanced_bone_age ? '✓' : ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;
  return (
    <div>
      <h3 style={{ color: '#1565c0' }}>{data.atlas}</h3>
      <p><strong>Pathway:</strong> {data.pathway}</p>
      <div style={{ background: '#e3f2fd', borderRadius: 8, padding: 14, marginBottom: 16 }}>
        <strong>Shared Mechanism:</strong> {data.shared_mechanism}
      </div>

      <h4>Gene Summaries</h4>
      {data.genes && Object.entries(data.genes).map(([g, info]) => (
        <div key={g} style={{ borderLeft: `4px solid ${GENE_COLORS[g] || '#333'}`, paddingLeft: 12, marginBottom: 14 }}>
          <strong style={{ color: GENE_COLORS[g] || '#333' }}>{g}</strong> — {info.locus} — {info.protein_size} — {info.inheritance?.split(' ')[0]}
          <div style={{ fontSize: 12, color: '#444', marginTop: 4 }}>{info.pathognomonic?.substring(0, 250)}…</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, marginTop: 6 }}>
            {info.critical_flags?.map(f => (
              <span key={f} style={{ background: GENE_COLORS[g] || '#333', color: '#fff', borderRadius: 3, padding: '1px 6px', fontSize: 10 }}>{f}</span>
            ))}
          </div>
        </div>
      ))}

      <h4>Glossary</h4>
      {data.glossary && Object.entries(data.glossary).map(([term, def]) => (
        <div key={term} style={{ marginBottom: 10 }}>
          <strong>{term}:</strong> <span style={{ color: '#444' }}>{def}</span>
        </div>
      ))}

      <h4>Surveillance Protocols</h4>
      {data.surveillance_protocols && Object.entries(data.surveillance_protocols).map(([gene, proto]) => (
        <div key={gene} style={{ marginBottom: 8, borderLeft: `3px solid ${GENE_COLORS[gene] || '#ccc'}`, paddingLeft: 10 }}>
          <strong style={{ color: GENE_COLORS[gene] || '#333' }}>{gene}:</strong> <span style={{ color: '#444' }}>{proto}</span>
        </div>
      ))}
    </div>
  );
}

export default function HereditarySkeletalDysplasiaAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    const base = `${API}/api/${SLUG}`;
    Promise.all([
      fetch(`${base}/overview`).then(r => r.json()),
      fetch(`${base}/breakdown`).then(r => r.json()),
      fetch(`${base}/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(df);
    }).catch(e => setErr(String(e)));
  }, []);

  return (
    <div style={{ fontFamily: 'sans-serif', padding: '16px 24px', maxWidth: 1400, margin: '0 auto' }}>
      <h1 style={{ color: '#1565c0', marginBottom: 4 }}>🦴 Hereditary Skeletal Dysplasia Atlas</h1>
      <p style={{ color: '#555', marginBottom: 16 }}>
        Complete 8-Gene Hereditary Skeletal Dysplasia Reference — FGFR3 · COL2A1 · EXT1 · EXT2 · COMP · SLC26A2 · TRPV4 · ACAN
        (320 patients, seeds 2038-2045)
      </p>
      {err && <ErrBox msg={err} />}

      <div style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            style={{
              padding: '8px 18px', borderRadius: 6, border: 'none', cursor: 'pointer',
              background: tab === t ? '#1565c0' : '#e3f2fd',
              color: tab === t ? '#fff' : '#1565c0',
              fontWeight: tab === t ? 700 : 400,
            }}
          >{t}</button>
        ))}
      </div>

      {tab === 'Overview'      && <OverviewTab data={overview} />}
      {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
      {tab === 'Clinical Atlas'&& <ClinicalAtlasTab data={breakdown} />}
      {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
    </div>
  );
}
