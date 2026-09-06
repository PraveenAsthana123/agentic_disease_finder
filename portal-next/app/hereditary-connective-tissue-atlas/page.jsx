'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  FBN1:   '#1565c0',  // deep blue — Marfan syndrome, fibrillin-1
  TGFBR1: '#b71c1c',  // deep red — LDS1, TGF-β receptor 1
  TGFBR2: '#c62828',  // crimson — LDS2, TGF-β receptor 2
  COL3A1: '#4a148c',  // deep purple — vascular EDS, most lethal
  COL5A1: '#2e7d32',  // deep green — classical EDS, skin hyperextensibility
  ACTA2:  '#e65100',  // deep orange — HTAD, livedo reticularis pathognomonic
  MYH11:  '#f57f17',  // amber — HTAD + PDA + MVP triad
  SMAD3:  '#006064',  // dark cyan — LDS3/AOS, early OA + aneurysm
};

const GENE_DISEASE = {
  FBN1:   'Marfan Syndrome AD Aortic Root + Ectopia Lentis — Beta-Blocker + Losartan — Surgery ≥50 mm — No Contact Sports Lifelong — Arm Span > Height — Ghent 2010',
  TGFBR1: 'LDS1 AD Bifid Uvula + Hypertelorism + Tortuosity TRIAD — Surgery ≥42 mm NOT ≥50 mm — Head-to-Pelvis MRA Mandatory — Losartan PRIMARY — Craniosynostosis 10-15%',
  TGFBR2: 'LDS2 AD Club Foot + Skin Findings — Surgery ≥42 mm — Head-to-Pelvis MRA — Losartan STOP Conception Teratogenic — Lynch Somatic TGFBR2 Different',
  COL3A1: 'vEDS AD Most Lethal EDS — Spontaneous Arterial Rupture Celiac/Mesenteric — Bowel Perforation — NO Elective Surgery — NO Colonoscopy — Celiprolol BBEST RCT Evidence',
  COL5A1: 'Classical EDS AD Skin Hyperextensibility + Atrophic Scarring + Joint Hypermobility — NOT Vascular Danger — Physiotherapy + Bracing — Wound Care Steri-Strips',
  ACTA2:  'HTAD AD Livedo Reticularis + Iris Flocculi PATHOGNOMONIC Arg179 — Moyamoya Screen — Premature CAD Statin — Surgery ≥45-50 mm — PDA History Ask',
  MYH11:  'HTAD+PDA AD Aortic Aneurysm + Patent Ductus Arteriosus + MVP TRIAD — PDA Often Repaired Infancy Ask History — Surgery ≥50 mm — Annual TTE',
  SMAD3:  'LDS3/AOS AD Early OA 20s-30s + Aortic Aneurysm PATHOGNOMONIC — Surgery ≥42 mm — Losartan PRIMARY + Beta-Blocker — NSAIDs OA — Bifid Uvula 50% Only',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /NO\s+ELECTIVE|NO\s+COLONOSCOPY|ABSOLUTE|CONTRAINDICATED|NEVER|STOP|FATAL|LETHAL|PATHOGNOMONIC|TERATOGENIC|MANDATORY|AVOID|PROHIBITED/i.test(text);
  const isWarning = /SURGERY|THRESHOLD|WARNING|MONITOR|SCREEN|ANNUAL|REQUIRED|PROTOCOL|TRIAL|AGGRESSIVE|TRIAD|NOT\s+≥50|DISTINGUISH|ASK|CASCADE/i.test(text);
  const bg = isCI ? '#b71c1c' : isWarning ? '#e65100' : '#1565c0';
  return (
    <div style={{
      background: bg, color: '#fff', borderRadius: 6, padding: '6px 12px',
      marginBottom: 8, fontSize: 13, lineHeight: 1.4,
    }}>
      {text}
    </div>
  );
}

function OverviewTab({ data }) {
  if (!data) return <Loading />;
  const { aggregate_stats: s, top_alerts, genes } = data;

  const statRows = [
    ['Total patients', s.total_patients],
    ['Mean age at diagnosis (all genes)', `${s.mean_dx_age_years} yr`],
    ['Mean diagnostic delay (all genes)', `${s.mean_dx_delay_months} mo`],
    ['Aortic surgery performed (any gene)', `${s.aortic_surgery_pct}%`],
    ['Aortic dissection event (any gene)', `${s.aortic_dissection_pct}%`],
    ['Beta-blocker prescribed (any gene)', `${s.beta_blocker_pct}%`],
    ['Losartan prescribed (any gene)', `${s.losartan_pct}%`],
    ['Sport restriction documented', `${s.sport_restriction_pct}%`],
    ['Cascade genetic testing performed', `${s.cascade_tested_pct}%`],
    ['De novo variant (no family history)', `${s.de_novo_pct}%`],
    // FBN1 Marfan
    ['FBN1 — ectopia lentis (superotemporal)', `${s.fbn1_ectopia_lentis_pct}%`],
    ['FBN1 — aortic surgery performed', `${s.fbn1_aortic_surgery_pct}%`],
    ['FBN1 — beta-blocker prescribed', `${s.fbn1_beta_blocker_pct}%`],
    ['FBN1 — losartan prescribed', `${s.fbn1_losartan_pct}%`],
    ['FBN1 — sport restriction documented', `${s.fbn1_sport_restriction_pct}%`],
    ['FBN1 — joint hypermobility', `${s.fbn1_joint_hypermobility_pct}%`],
    // TGFBR1 LDS1
    ['TGFBR1 — bifid uvula present', `${s.tgfbr1_bifid_uvula_pct}%`],
    ['TGFBR1 — hypertelorism present', `${s.tgfbr1_hypertelorism_pct}%`],
    ['TGFBR1 — arterial tortuosity', `${s.tgfbr1_arterial_tortuosity_pct}%`],
    ['TGFBR1 — aortic surgery (at ≥42 mm)', `${s.tgfbr1_aortic_surgery_pct}%`],
    ['TGFBR1 — losartan prescribed', `${s.tgfbr1_losartan_pct}%`],
    // TGFBR2 LDS2
    ['TGFBR2 — bifid uvula present', `${s.tgfbr2_bifid_uvula_pct}%`],
    ['TGFBR2 — arterial tortuosity', `${s.tgfbr2_arterial_tortuosity_pct}%`],
    ['TGFBR2 — aortic surgery (at ≥42 mm)', `${s.tgfbr2_aortic_surgery_pct}%`],
    ['TGFBR2 — joint hypermobility', `${s.tgfbr2_joint_hypermobility_pct}%`],
    // COL3A1 vEDS
    ['COL3A1 — spontaneous arterial rupture', `${s.col3a1_arterial_rupture_pct}%`],
    ['COL3A1 — bowel perforation', `${s.col3a1_bowel_perforation_pct}%`],
    ['COL3A1 — celiprolol prescribed', `${s.col3a1_celiprolol_pct}%`],
    ['COL3A1 — aortic dissection', `${s.col3a1_aortic_dissection_pct}%`],
    ['COL3A1 — skin hyperextensibility (minimal)', `${s.col3a1_skin_hyperextensibility_pct}%`],
    // COL5A1 cEDS
    ['COL5A1 — skin hyperextensibility', `${s.col5a1_skin_hyperextensibility_pct}%`],
    ['COL5A1 — widened atrophic scarring', `${s.col5a1_atrophic_scarring_pct}%`],
    ['COL5A1 — joint hypermobility (Beighton ≥5)', `${s.col5a1_joint_hypermobility_pct}%`],
    ['COL5A1 — aortic surgery (uncommon)', `${s.col5a1_aortic_surgery_pct}%`],
    // ACTA2 HTAD
    ['ACTA2 — livedo reticularis (Arg179)', `${s.acta2_livedo_reticularis_pct}%`],
    ['ACTA2 — iris flocculi (Arg179 pathognomonic)', `${s.acta2_iris_flocculi_pct}%`],
    ['ACTA2 — Moyamoya cerebrovascular', `${s.acta2_moyamoya_pct}%`],
    ['ACTA2 — aortic surgery performed', `${s.acta2_aortic_surgery_pct}%`],
    ['ACTA2 — PDA history', `${s.acta2_pda_history_pct}%`],
    // MYH11 HTAD+PDA
    ['MYH11 — patent ductus arteriosus (PDA) history', `${s.myh11_pda_history_pct}%`],
    ['MYH11 — mitral valve prolapse (MVP)', `${s.myh11_mvp_pct}%`],
    ['MYH11 — aortic surgery performed', `${s.myh11_aortic_surgery_pct}%`],
    ['MYH11 — beta-blocker prescribed', `${s.myh11_beta_blocker_pct}%`],
    // SMAD3 LDS3/AOS
    ['SMAD3 — early-onset osteoarthritis (20s–30s)', `${s.smad3_early_oa_pct}%`],
    ['SMAD3 — bifid uvula (~50% only)', `${s.smad3_bifid_uvula_pct}%`],
    ['SMAD3 — arterial tortuosity', `${s.smad3_arterial_tortuosity_pct}%`],
    ['SMAD3 — aortic surgery (at ≥42 mm)', `${s.smad3_aortic_surgery_pct}%`],
    ['SMAD3 — losartan prescribed', `${s.smad3_losartan_pct}%`],
  ];

  return (
    <div style={{ padding: '1rem' }}>
      <h2 style={{ marginBottom: '0.5rem' }}>
        Hereditary Connective Tissue Disorders Atlas — 320 Patients, 8 Genes, Seeds 1542–1549
      </h2>
      <p style={{ color: '#555', marginBottom: '1rem', fontSize: 14 }}>
        FBN1 (Marfan) · TGFBR1 (LDS1) · TGFBR2 (LDS2) · COL3A1 (vEDS) · COL5A1 (Classical EDS) · ACTA2 (HTAD) · MYH11 (HTAD+PDA) · SMAD3 (LDS3/AOS)
      </p>

      <h3>Top Clinical Alerts</h3>
      <div style={{ marginBottom: '1.5rem' }}>
        {top_alerts && top_alerts.map((a, i) => (
          <div key={i} style={{ marginBottom: 4 }}>
            <span style={{ background: GENE_COLORS[a.gene] || '#333', color: '#fff', borderRadius: 4, padding: '2px 7px', fontSize: 12, marginRight: 8 }}>{a.gene}</span>
            <AlertBadge text={a.alert} />
          </div>
        ))}
      </div>

      <h3>Aggregate Statistics</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '1.5rem', fontSize: 13 }}>
        <tbody>
          {statRows.map(([label, val], i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff' }}>
              <td style={{ padding: '5px 10px', color: '#444', width: '60%' }}>{label}</td>
              <td style={{ padding: '5px 10px', fontWeight: 600 }}>{val}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h3>Gene Summary</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '1rem' }}>
        {genes && genes.map(g => (
          <div key={g.gene} style={{
            border: `2px solid ${GENE_COLORS[g.gene] || '#ccc'}`,
            borderRadius: 8, padding: '0.75rem', background: '#fff',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
              <span style={{ background: GENE_COLORS[g.gene], color: '#fff', borderRadius: 4, padding: '2px 8px', fontWeight: 700, fontSize: 14 }}>{g.gene}</span>
              <span style={{ fontSize: 12, color: '#555' }}>{g.aa} · {g.kDa} · {g.locus}</span>
            </div>
            <div style={{ fontSize: 12, color: '#333', marginBottom: 6, lineHeight: 1.4 }}>{GENE_DISEASE[g.gene]}</div>
            <div style={{ fontSize: 12, color: '#555' }}>
              <b>OMIM:</b> Gene {g.omim_gene} · Disease {g.omim_disease} · <b>n=</b>{g.n_patients} pts<br/>
              <b>Inheritance:</b> {g.inheritance}<br/>
              <b>Mean age dx:</b> {g.mean_dx_age} yr · <b>Dx delay:</b> {g.mean_dx_delay_months} mo · <b>Root:</b> {g.mean_aortic_root_mm} mm<br/>
              <b>Surgery:</b> {g.aortic_surgery_pct}% · <b>Dissection:</b> {g.aortic_dissection_pct}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function GeneTableTab({ data }) {
  const [selGene, setSelGene] = useState(null);
  if (!data) return <Loading />;
  const genes = Object.keys(data);
  const gd = selGene ? data[selGene] : null;

  return (
    <div style={{ padding: '1rem' }}>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: '1rem' }}>
        {genes.map(g => (
          <button key={g} onClick={() => setSelGene(g === selGene ? null : g)}
            style={{
              background: selGene === g ? (GENE_COLORS[g] || '#333') : '#eee',
              color: selGene === g ? '#fff' : '#333',
              border: 'none', borderRadius: 6, padding: '6px 14px', cursor: 'pointer', fontWeight: 600,
            }}>
            {g}
          </button>
        ))}
      </div>

      {!gd ? (
        <div style={{ color: '#555' }}>Select a gene to view its detailed breakdown.</div>
      ) : (
        <div>
          <h3 style={{ color: GENE_COLORS[gd.gene] }}>{gd.gene} — {gd.n_patients} Patients</h3>
          <p style={{ fontSize: 13, color: '#444', marginBottom: 8 }}>{gd.protein}</p>

          <h4>Etiologies</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '1rem', fontSize: 12 }}>
            <thead>
              <tr style={{ background: GENE_COLORS[gd.gene], color: '#fff' }}>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Etiology</th>
                <th style={{ padding: '6px 8px' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(gd.etiologies).map(([etiol, cnt], i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                  <td style={{ padding: '5px 8px', fontSize: 11 }}>{etiol}</td>
                  <td style={{ padding: '5px 8px', textAlign: 'center', fontWeight: 600 }}>{cnt}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
            <div>
              <h4>Age at Diagnosis Distribution</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <tbody>
                  {Object.entries(gd.age_at_diagnosis_distribution).map(([bucket, cnt], i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                      <td style={{ padding: '4px 8px' }}>{bucket} yr</td>
                      <td style={{ padding: '4px 8px', fontWeight: 600 }}>{cnt}</td>
                      <td style={{ padding: '4px 8px' }}>
                        <div style={{ background: GENE_COLORS[gd.gene], height: 10, width: `${cnt * 8}px`, borderRadius: 3 }} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div>
              <h4>Aortic Root at Diagnosis (mm)</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <tbody>
                  {Object.entries(gd.aortic_root_distribution_mm).map(([bucket, cnt], i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                      <td style={{ padding: '4px 8px' }}>{bucket} mm</td>
                      <td style={{ padding: '4px 8px', fontWeight: 600 }}>{cnt}</td>
                      <td style={{ padding: '4px 8px' }}>
                        <div style={{ background: GENE_COLORS[gd.gene], height: 10, width: `${cnt * 8}px`, borderRadius: 3 }} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <h4>Clinical Statistics</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginBottom: '1rem' }}>
            <tbody>
              {Object.entries(gd.stats).map(([k, v], i) => (
                <tr key={i} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                  <td style={{ padding: '4px 8px', color: '#444' }}>{k.replace(/_/g, ' ')}</td>
                  <td style={{ padding: '4px 8px', fontWeight: 600 }}>{typeof v === 'number' ? v : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h4>Sample Patients (first 10)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr style={{ background: GENE_COLORS[gd.gene], color: '#fff' }}>
                  {['ID', 'Sex', 'Age Dx', 'Dx Delay', 'Root mm', 'Surgery', 'Dissection', 'Beta-Blk', 'Losartan', 'De Novo'].map(h => (
                    <th key={h} style={{ padding: '5px 6px' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {gd.patients && gd.patients.map((p, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                    <td style={{ padding: '4px 6px' }}>{p.patient_id}</td>
                    <td style={{ padding: '4px 6px' }}>{p.sex}</td>
                    <td style={{ padding: '4px 6px' }}>{p.age_at_diagnosis} yr</td>
                    <td style={{ padding: '4px 6px' }}>{p.dx_delay_months} mo</td>
                    <td style={{ padding: '4px 6px' }}>{p.aortic_root_mm}</td>
                    <td style={{ padding: '4px 6px' }}>{p.aortic_surgery ? '✓' : '–'}</td>
                    <td style={{ padding: '4px 6px' }}>{p.aortic_dissection ? '⚠' : '–'}</td>
                    <td style={{ padding: '4px 6px' }}>{p.beta_blocker ? '✓' : '–'}</td>
                    <td style={{ padding: '4px 6px' }}>{p.losartan ? '✓' : '–'}</td>
                    <td style={{ padding: '4px 6px' }}>{p.de_novo ? '✓' : '–'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function ClinicalAtlasTab({ data }) {
  if (!data) return <Loading />;

  return (
    <div style={{ padding: '1rem' }}>
      <h3>Clinical Atlas — Key Pharmacological &amp; Management Distinctions</h3>
      <p style={{ fontSize: 13, color: '#555', marginBottom: '1rem' }}>
        Critical gene-by-gene management rules and safety alerts for hereditary connective tissue disorders.
      </p>
      {Object.values(data).map(gd => (
        <div key={gd.gene} style={{ marginBottom: '2rem', borderLeft: `4px solid ${GENE_COLORS[gd.gene] || '#ccc'}`, paddingLeft: '1rem' }}>
          <h4 style={{ color: GENE_COLORS[gd.gene], margin: '0 0 0.5rem 0' }}>
            {gd.gene} — {gd.protein.split('—')[0].trim()}
          </h4>
          <p style={{ fontSize: 12, color: '#444', marginBottom: '0.75rem', lineHeight: 1.5 }}>
            <b>Locus:</b> {gd.gene} · <b>Inheritance:</b> {gd.stats && (
              `Mean dx age: ${gd.stats.mean_dx_age} yr · Mean aortic root: ${gd.stats.mean_aortic_root_mm} mm`
            )}
          </p>
          {gd.key_alerts && gd.key_alerts.map((alert, i) => (
            <AlertBadge key={i} text={alert} />
          ))}
        </div>
      ))}
    </div>
  );
}

function DefinitionsTab({ data }) {
  if (!data) return <Loading />;

  return (
    <div style={{ padding: '1rem' }}>
      <h3>Definitions &amp; Concepts — Hereditary Connective Tissue Disorders Atlas</h3>

      <h4>Core Concepts</h4>
      {Object.entries(data.concepts || {}).map(([term, def], i) => (
        <div key={i} style={{ marginBottom: '0.75rem', background: '#f5f5f5', borderRadius: 6, padding: '0.6rem 0.9rem' }}>
          <div style={{ fontWeight: 700, color: '#1565c0', marginBottom: 3 }}>{term.replace(/_/g, ' ')}</div>
          <div style={{ fontSize: 13, color: '#444', lineHeight: 1.5 }}>{def}</div>
        </div>
      ))}

      <h4 style={{ marginTop: '1.5rem' }}>Pharmacological Distinctions</h4>
      {(data.pharmacological_distinctions || []).map((rule, i) => (
        <AlertBadge key={i} text={rule} />
      ))}

      <h4 style={{ marginTop: '1.5rem' }}>Key Standards &amp; References</h4>
      <ul style={{ fontSize: 13, lineHeight: 1.7, color: '#444' }}>
        {(data.key_standards || []).map((s, i) => <li key={i}>{s}</li>)}
      </ul>
    </div>
  );
}

export default function HereditaryConnectiveTissueAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-connective-tissue-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
    fetch(`${API}/api/hereditary-connective-tissue-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/hereditary-connective-tissue-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  if (error) return <div style={{ padding: '2rem', color: '#b71c1c' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ background: '#1565c0', color: '#fff', padding: '1rem 1.5rem' }}>
        <h1 style={{ margin: 0, fontSize: 20 }}>
          &#x1f9ec; Hereditary Connective Tissue Disorders Atlas — Complete 8-Gene Reference
        </h1>
        <p style={{ margin: '4px 0 0', fontSize: 13, opacity: 0.85 }}>
          FBN1 · TGFBR1 · TGFBR2 · COL3A1 · COL5A1 · ACTA2 · MYH11 · SMAD3 — 320 Patients (8×40, Seeds 1542–1549)
        </p>
      </div>

      <div style={{ display: 'flex', borderBottom: '2px solid #e0e0e0', background: '#fafafa' }}>
        {TABS.map(t => (
          <button key={t} onClick={() => setTab(t)}
            style={{
              padding: '10px 20px', border: 'none', borderBottom: tab === t ? '3px solid #1565c0' : '3px solid transparent',
              background: 'transparent', cursor: 'pointer', fontWeight: tab === t ? 700 : 400,
              color: tab === t ? '#1565c0' : '#555', fontSize: 14,
            }}>
            {t}
          </button>
        ))}
      </div>

      <div style={{ minHeight: 400 }}>
        {tab === 'Overview'      && <OverviewTab data={overview} />}
        {tab === 'Gene Table'    && <GeneTableTab data={breakdown} />}
        {tab === 'Clinical Atlas' && <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions'   && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
