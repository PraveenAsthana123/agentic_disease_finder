'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Gene Table', 'Clinical Atlas', 'Definitions'];

const GENE_COLORS = {
  MEFV:      '#1565c0',  // deep blue — FMF, most common periodic fever
  TNFRSF1A:  '#b71c1c',  // deep red — TRAPS, TNF receptor, prolonged attacks
  MVK:       '#4a148c',  // deep purple — HIDS, mevalonate kinase
  NLRP3:     '#e65100',  // deep orange — CAPS, inflammasome GOF
  NOD2:      '#2e7d32',  // deep green — Blau syndrome, granulomatous triad
  IL1RN:     '#006064',  // dark cyan — DIRA, IL-1Ra deficiency
  ADA2:      '#f57f17',  // amber — DADA2, childhood stroke
  PSTPIP1:   '#880e4f',  // deep pink/purple — PAPA, pyoderma gangrenosum
};

const GENE_DISEASE = {
  MEFV:      'FMF AR/AD Colchicine FIRST LINE — SAA Amyloidosis Prevention — IL-1 for Resistance — Peritonitis Most Common — M694V Highest Risk — Colchicine SAFE Pregnancy',
  TNFRSF1A:  'TRAPS AD Fever >7 Days — Migratory Myalgia + Periorbital Oedema PATHOGNOMONIC — Etanercept NOT Infliximab — R92Q Low Penetrance — SAA Monitor High Penetrance',
  MVK:       'HIDS/MKD AR Cervical Lymphadenopathy PATHOGNOMONIC >95% — IgD >100 IU/mL — Canakinumab FDA 2021 — Urine MMA During Attack — Enzyme Assay Mandatory',
  NLRP3:     'CAPS AD GOF Canakinumab FIRST LINE — FCAS Cold-Triggered — MWS Hearing Loss Progressive — NOMID Neonatal Meningitis + Blindness + Deafness Most Severe — Somatic Mosaic 35%',
  NOD2:      'Blau Syndrome AD GOF TRIAD: Skin + Joint + Eye Granulomatous PATHOGNOMONIC — Onset <4 Years — Adalimumab for Uveitis — Band Keratopathy PATHOGNOMONIC — NOT Crohn Variants',
  IL1RN:     'DIRA AR Anakinra CURATIVE — Neonatal Pustulosis + Periostitis PATHOGNOMONIC — NO Fever Paradox — STOP = Relapse 24–72h — Continuous LIFELONG — Newfoundland Founder',
  ADA2:      'DADA2 AR TNF Blockers PREVENT STROKE — Childhood Lacunar Infarcts — Livedo Racemosa — Enzyme Activity Assay < 2 nmol/hr/mL — HSCT Haematological Only — NOT ADA1-SCID',
  PSTPIP1:   'PAPA AD TRIAD: Pyogenic Arthritis + Pyoderma Gangrenosum + Severe Acne — NEVER Debride PG (Pathergy) — IL-1/TNF Biologics — A230T Most Common — Joint Biopsy Sterile',
};

function Loading() {
  return <div style={{ padding: '2rem', color: '#666' }}>Loading…</div>;
}

function AlertBadge({ text }) {
  const isCI = /NO\s+ELECTIVE|ABSOLUTE|CONTRAINDICATED|NEVER|STOP|FATAL|LETHAL|PATHOGNOMONIC|TERATOGENIC|CURATIVE|PREVENT|NOT\s+ADA1|NOT\s+INFLIXIMAB|NOT\s+DEBRIDE|AVOID/i.test(text);
  const isWarning = /MONITOR|SCREEN|ANNUAL|REQUIRED|MANDATORY|PROTOCOL|FIRST\s+LINE|FIRST-LINE|CONTINUOUS|LIFELONG|IMMEDIATELY|CASCADE|DISTINGUISH|START|ASSAY/i.test(text);
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
    ['Cascade genetic testing performed', `${s.cascade_tested_pct}%`],
    // MEFV FMF
    ['MEFV — colchicine prescribed', `${s.mefv_colchicine_pct}%`],
    ['MEFV — colchicine resistant (≥1 attack/month)', `${s.mefv_colchicine_resistant_pct}%`],
    ['MEFV — IL-1 biologic prescribed', `${s.mefv_il1_biologic_pct}%`],
    ['MEFV — SAA amyloidosis developed', `${s.mefv_amyloidosis_pct}%`],
    ['MEFV — peritonitis (most common attack feature)', `${s.mefv_peritonitis_pct}%`],
    // TNFRSF1A TRAPS
    ['TNFRSF1A — periorbital oedema present', `${s.tnfrsf1a_periorbital_oedema_pct}%`],
    ['TNFRSF1A — migratory myalgia present', `${s.tnfrsf1a_migratory_myalgia_pct}%`],
    ['TNFRSF1A — etanercept prescribed', `${s.tnfrsf1a_etanercept_pct}%`],
    ['TNFRSF1A — corticosteroid dependent', `${s.tnfrsf1a_corticosteroid_dependent_pct}%`],
    // MVK HIDS
    ['MVK — cervical lymphadenopathy (pathognomonic)', `${s.mvk_cervical_lymphadenopathy_pct}%`],
    ['MVK — IgD elevated (>100 IU/mL)', `${s.mvk_igd_elevated_pct}%`],
    ['MVK — canakinumab prescribed', `${s.mvk_canakinumab_pct}%`],
    // NLRP3 CAPS
    ['NLRP3 — canakinumab prescribed (first-line)', `${s.nlrp3_canakinumab_pct}%`],
    ['NLRP3 — urticarial rash (all CAPS)', `${s.nlrp3_urticarial_rash_pct}%`],
    ['NLRP3 — sensorineural hearing loss (MWS/NOMID)', `${s.nlrp3_snhl_pct}%`],
    // NOD2 Blau
    ['NOD2 — granulomatous uveitis present', `${s.nod2_granulomatous_uveitis_pct}%`],
    ['NOD2 — anti-TNF prescribed', `${s.nod2_anti_tnf_pct}%`],
    ['NOD2 — visual impairment (complication)', `${s.nod2_visual_impairment_pct}%`],
    // IL1RN DIRA
    ['IL1RN — anakinra prescribed (curative)', `${s.il1rn_anakinra_pct}%`],
    ['IL1RN — periostitis on X-ray (pathognomonic)', `${s.il1rn_periostitis_pct}%`],
    ['IL1RN — initially diagnosed as sepsis', `${s.il1rn_initial_sepsis_pct}%`],
    // ADA2 DADA2
    ['ADA2 — TNF blocker prescribed (stroke prevention)', `${s.ada2_tnf_blocker_pct}%`],
    ['ADA2 — ischaemic stroke occurred', `${s.ada2_ischaemic_stroke_pct}%`],
    ['ADA2 — livedo racemosa present', `${s.ada2_livedo_racemosa_pct}%`],
    ['ADA2 — HSCT performed (haematological)', `${s.ada2_hsct_pct}%`],
    // PSTPIP1 PAPA
    ['PSTPIP1 — pyogenic arthritis (sterile, destructive)', `${s.pstpip1_pyogenic_arthritis_pct}%`],
    ['PSTPIP1 — pyoderma gangrenosum present', `${s.pstpip1_pyoderma_gangrenosum_pct}%`],
    ['PSTPIP1 — IL-1 biologic prescribed', `${s.pstpip1_il1_biologic_pct}%`],
    ['PSTPIP1 — pathergy test positive', `${s.pstpip1_pathergy_positive_pct}%`],
  ];

  return (
    <div style={{ padding: '1rem' }}>
      <h2 style={{ marginBottom: '0.5rem' }}>
        Hereditary Autoinflammatory Disorders Atlas — 320 Patients, 8 Genes, Seeds 1550–1557
      </h2>
      <p style={{ color: '#555', marginBottom: '1rem', fontSize: 14 }}>
        MEFV (FMF) · TNFRSF1A (TRAPS) · MVK (HIDS/MKD) · NLRP3 (CAPS) · NOD2 (Blau) · IL1RN (DIRA) · ADA2 (DADA2) · PSTPIP1 (PAPA)
      </p>

      <h3>Top Clinical Alerts</h3>
      <div style={{ marginBottom: '1.5rem' }}>
        {top_alerts && top_alerts.map((a, i) => (
          <div key={i} style={{ marginBottom: 4, display: 'flex', alignItems: 'flex-start', gap: 8 }}>
            <span style={{ background: GENE_COLORS[a.gene] || '#333', color: '#fff', borderRadius: 4, padding: '2px 7px', fontSize: 12, whiteSpace: 'nowrap', marginTop: 2 }}>{a.gene}</span>
            <AlertBadge text={a.alert} />
          </div>
        ))}
      </div>

      <h3>Aggregate Statistics</h3>
      <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '1.5rem', fontSize: 13 }}>
        <tbody>
          {statRows.map(([label, val], i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? '#f5f5f5' : '#fff' }}>
              <td style={{ padding: '5px 10px', color: '#444', width: '65%' }}>{label}</td>
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
              <b>Mean age dx:</b> {g.mean_dx_age} yr · <b>Dx delay:</b> {g.mean_dx_delay_months} mo
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

          <h4>Etiologies (Variants)</h4>
          <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: '1rem', fontSize: 12 }}>
            <thead>
              <tr style={{ background: GENE_COLORS[gd.gene], color: '#fff' }}>
                <th style={{ padding: '6px 8px', textAlign: 'left' }}>Variant / Genotype</th>
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
              <h4>Diagnostic Delay Distribution</h4>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <tbody>
                  {Object.entries(gd.dx_delay_distribution || {}).map(([bucket, cnt], i) => (
                    <tr key={i} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                      <td style={{ padding: '4px 8px' }}>{bucket}</td>
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
                  <td style={{ padding: '4px 8px', fontWeight: 600 }}>{typeof v === 'number' ? (Number.isInteger(v) || v > 10 ? v : v + '%') : String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h4>Key Alerts</h4>
          {gd.key_alerts && gd.key_alerts.map((alert, i) => (
            <AlertBadge key={i} text={alert} />
          ))}

          <h4 style={{ marginTop: '1rem' }}>Sample Patients (first 10)</h4>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
              <thead>
                <tr style={{ background: GENE_COLORS[gd.gene], color: '#fff' }}>
                  <th style={{ padding: '5px 6px' }}>ID</th>
                  <th style={{ padding: '5px 6px' }}>Variant/Genotype</th>
                  <th style={{ padding: '5px 6px' }}>Age Dx</th>
                  <th style={{ padding: '5px 6px' }}>Delay (mo)</th>
                </tr>
              </thead>
              <tbody>
                {gd.patients && gd.patients.map((p, i) => (
                  <tr key={i} style={{ background: i % 2 === 0 ? '#f9f9f9' : '#fff' }}>
                    <td style={{ padding: '4px 6px' }}>{p.id}</td>
                    <td style={{ padding: '4px 6px' }}>{p.etiology}</td>
                    <td style={{ padding: '4px 6px' }}>{p.age_at_diagnosis} yr</td>
                    <td style={{ padding: '4px 6px' }}>{p.dx_delay_months} mo</td>
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
      <h3>Clinical Atlas — Key Management Distinctions by Gene</h3>
      <p style={{ fontSize: 13, color: '#555', marginBottom: '1rem' }}>
        Critical gene-by-gene management rules and safety alerts for hereditary autoinflammatory disorders.
        Distinguish periodic fever syndromes by attack duration, clinical features, and treatment response.
      </p>
      {Object.values(data).map(gd => (
        <div key={gd.gene} style={{ marginBottom: '2rem', borderLeft: `4px solid ${GENE_COLORS[gd.gene] || '#ccc'}`, paddingLeft: '1rem' }}>
          <h4 style={{ color: GENE_COLORS[gd.gene], margin: '0 0 0.5rem 0' }}>
            {gd.gene} — {gd.protein.split('—')[0].trim()}
          </h4>
          <p style={{ fontSize: 12, color: '#444', marginBottom: '0.75rem', lineHeight: 1.5 }}>
            {gd.stats && (
              `Mean dx age: ${gd.stats.mean_dx_age} yr · Mean diagnostic delay: ${gd.stats.mean_dx_delay_months} mo`
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
      <h3>Definitions &amp; Concepts — Hereditary Autoinflammatory Disorders Atlas</h3>

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

export default function HereditaryAutoinflammtoryAtlasPage() {
  const [tab, setTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/hereditary-autoinflammatory-atlas/overview`)
      .then(r => r.json()).then(setOverview).catch(e => setError(String(e)));
    fetch(`${API}/api/hereditary-autoinflammatory-atlas/breakdown`)
      .then(r => r.json()).then(setBreakdown).catch(() => {});
    fetch(`${API}/api/hereditary-autoinflammatory-atlas/definitions`)
      .then(r => r.json()).then(setDefinitions).catch(() => {});
  }, []);

  if (error) return <div style={{ padding: '2rem', color: '#b71c1c' }}>Error: {error}</div>;

  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', maxWidth: 1400, margin: '0 auto' }}>
      <div style={{ background: '#1565c0', color: '#fff', padding: '1rem 1.5rem' }}>
        <h1 style={{ margin: 0, fontSize: 20 }}>
          &#x1f9ec; Hereditary Autoinflammatory Disorders Atlas — Complete 8-Gene Reference
        </h1>
        <p style={{ margin: '4px 0 0', fontSize: 13, opacity: 0.85 }}>
          MEFV · TNFRSF1A · MVK · NLRP3 · NOD2 · IL1RN · ADA2 · PSTPIP1 — 320 Patients (8×40, Seeds 1550–1557)
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
        {tab === 'Overview'       && <OverviewTab data={overview} />}
        {tab === 'Gene Table'     && <GeneTableTab data={breakdown} />}
        {tab === 'Clinical Atlas'  && <ClinicalAtlasTab data={breakdown} />}
        {tab === 'Definitions'    && <DefinitionsTab data={definitions} />}
      </div>
    </div>
  );
}
