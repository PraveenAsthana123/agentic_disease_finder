'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'CPLANE Complex Pearls', 'Definitions'];

// JBTS33 colour scheme — CPLANE1 / Ciliogenesis PCP Effector / BB Docking / No MKS Tier
// Forest green for CPLANE complex; teal for BB docking; amber for PCP/polydactyly
const ACCENT   = '#1b5e20';   // dark forest green — CPLANE1 ciliogenesis identity
const ACCENT2  = '#2e7d32';   // forest green — CPLANE complex / BB docking
const ACCENT3  = '#00695c';   // dark teal — PCP effector / INTURNED-FUZ axis
const ACCENT4  = '#1565c0';   // royal blue — no MKS tier / all liveborn
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#e65100';   // deep orange — polydactyly enriched / PCP alert
const ACCENT7  = '#4a148c';   // deep purple — renal NPHP-like / retinal
const ACCENT8  = '#0277bd';   // mid blue — cilia length / BB docking angle
const ACCENT9  = '#558b2f';   // olive green — MTS / cerebellar
const ACCENT10 = '#795548';   // brown — intellectual disability

const SEED = 485;
const N_COHORT = 40;

function KPI({ label, value, color }) {
  return (
    <div className="col-6 col-md-4 col-lg-2 mb-3">
      <div className="card h-100 shadow-sm text-center">
        <div className="card-body py-2 px-1">
          <div className="fw-bold fs-5" style={{ color }}>{value}</div>
          <div className="text-muted small">{label}</div>
        </div>
      </div>
    </div>
  );
}

function Alert({ color, children }) {
  return (
    <div className="alert mb-3" style={{ background: color + '18', borderLeft: `4px solid ${color}`, borderRadius: 6 }}>
      {children}
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color, borderBottom: `2px solid ${color}`, paddingBottom: 4 }}>{title}</h6>
      {children}
    </div>
  );
}

function DataTable({ headers, rows, accent }) {
  return (
    <div className="table-responsive mb-3">
      <table className="table table-sm table-hover" style={{ fontSize: 12 }}>
        <thead style={{ background: accent + '22' }}>
          <tr>{headers.map(h => <th key={h} style={{ color: accent }}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function JBTS33Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);
  const [loading, setLoading]     = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/jbts33/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts33/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts33/definitions`).then(r => r.json()),
    ]).then(([ov, bk, df]) => {
      setOverview(ov); setBreakdown(bk); setDefs(df);
      setLoading(false);
    }).catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-4 text-center"><div className="spinner-border" style={{ color: ACCENT }} /></div>;
  if (error)   return <div className="p-4 alert alert-danger">API error: {error}</div>;

  const kpis = overview?.key_kpis || {};

  return (
    <div className="container-fluid py-3" style={{ maxWidth: 1200 }}>
      {/* Header */}
      <div className="mb-3 p-3 rounded" style={{ background: ACCENT + '12', borderLeft: `5px solid ${ACCENT}` }}>
        <div className="d-flex align-items-center gap-2 flex-wrap">
          <span style={{ fontSize: 28 }}>🧬</span>
          <div>
            <h4 className="mb-0 fw-bold" style={{ color: ACCENT }}>
              JBTS33 — CPLANE1 Joubert Syndrome Type 33
            </h4>
            <div className="text-muted small">
              CPLANE1 (CFAP126/FLTP) · Ciliogenesis PCP Effector · BB Docking · No MKS Tier ·
              16q24.1 · OMIM <span style={{ color: ACCENT2 }}>*614571 / #617409</span> ·
              Polydactyly ~24% (PCP-enriched) ·
              {N_COHORT}-patient cohort · seed {SEED}
            </div>
          </div>
        </div>
      </div>

      {/* PCP / polydactyly alert */}
      <Alert color={ACCENT6}>
        <strong style={{ color: ACCENT6 }}>⚠ POLYDACTYLY ENRICHED IN JBTS33 (~24%):</strong>&nbsp;
        CPLANE1 is a Planar Cell Polarity (PCP) effector — mispositioned basal body → aberrant GLI3FL/GLI3R ratio in
        limb bud → postaxial polydactyly. Rate is higher than average JBTS (~18%). NORMAL OFC (no microcephaly)
        distinguishes JBTS33 from JBTS32/KIF14. No MKS tier — all patients liveborn.
      </Alert>

      {/* Nav tabs */}
      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={t} className="nav-item">
            <button
              className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
              style={tab === i ? { color: ACCENT, borderBottomColor: ACCENT } : {}}
              onClick={() => setTab(i)}
            >{t}</button>
          </li>
        ))}
      </ul>

      {/* ── Tab 0: Overview ── */}
      {tab === 0 && overview && (
        <div>
          {/* KPI row */}
          <div className="row mb-3">
            <KPI label="MTS (all JBTS33)" value="100%" color={ACCENT9} />
            <KPI label="Normal OFC" value="100%" color={ACCENT4} />
            <KPI label="Cerebellar Ataxia" value={kpis.cerebellar_ataxia_pct} color={ACCENT9} />
            <KPI label="Neonatal Hypotonia" value={kpis.neonatal_hypotonia_pct} color={ACCENT2} />
            <KPI label="Intellectual Disability" value={kpis.intellectual_disability} color={ACCENT10} />
            <KPI label="Oculomotor Apraxia" value={kpis.oculomotor_apraxia_pct} color={ACCENT8} />
          </div>
          <div className="row mb-3">
            <KPI label="Breathing Dysregulation" value={kpis.breathing_dysreg_pct} color={ACCENT3} />
            <KPI label="Retinal Rod-Cone" value={kpis.retinal_pct} color={ACCENT7} />
            <KPI label="Renal NPHP-like" value={kpis.renal_pct} color={ACCENT7} />
            <KPI label="Hepatic Mild" value={kpis.hepatic_pct} color={ACCENT5} />
            <KPI label="Polydactyly Postaxial" value={kpis.polydactyly_pct} color={ACCENT6} />
            <KPI label="No MKS / All Liveborn" value="100%" color={ACCENT4} />
          </div>

          {/* Gene card */}
          <Section title="Gene: CPLANE1 (*614571)" color={ACCENT}>
            <div className="row">
              <div className="col-md-6">
                <DataTable
                  accent={ACCENT}
                  headers={['Property', 'Value']}
                  rows={[
                    ['Gene', 'CPLANE1 (Ciliogenesis and Planar Cell Polarity Effector 1)'],
                    ['Aliases', 'CFAP126, FLTP (Flattop)'],
                    ['OMIM Gene', '*614571'],
                    ['OMIM Disease', '#617409 (JBTS33)'],
                    ['Chromosome', '16q24.1'],
                    ['Protein length', '~1,373 aa'],
                    ['Protein class', 'Cytoplasmic PCP effector / BB-docking scaffold'],
                    ['Inheritance', 'Autosomal Recessive — biallelic hypomorphic LOF'],
                  ]}
                />
              </div>
              <div className="col-md-6">
                <DataTable
                  accent={ACCENT3}
                  headers={['Domain', 'Residues', 'Key Function']}
                  rows={[
                    ['N-term IDR / INTU-binding', 'aa 1–180', 'CPLANE complex nucleation; INTURNED docking stub'],
                    ['CC1 / FUZ-binding', 'aa 181–430', 'FUZ interaction; homodimerisation; BB-trafficking initiation'],
                    ['Central IDR / linker', 'aa 431–760', 'PCP-switch phosphorylation hub (CK1δ/ε, CDK5); conformational regulation'],
                    ['WD40-like β-propeller', 'aa 761–1,110', 'BB-docking anchor; mother centriole appendage contact; variant hotspot'],
                    ['C-term CC / membrane anchor', 'aa 1,111–1,373', 'PI4P membrane binding; RAB11/EHD vesicle contact; apical targeting'],
                  ]}
                />
              </div>
            </div>
          </Section>

          {/* CPLANE complex */}
          <Section title="CPLANE Complex — CPLANE1 + INTURNED + FUZZY" color={ACCENT3}>
            <div className="row">
              {Object.entries(overview.cplane_complex || {}).map(([member, desc]) => (
                <div key={member} className="col-md-4 mb-2">
                  <div className="p-2 rounded" style={{ background: ACCENT3 + '12', borderLeft: `3px solid ${ACCENT3}` }}>
                    <div className="fw-bold small" style={{ color: ACCENT3 }}>{member}</div>
                    <div className="small text-muted">{desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </Section>

          {/* CPLANE1 pathway */}
          <Section title="CPLANE1 Pathogenic Mechanism" color={ACCENT2}>
            <div className="row">
              <div className="col-md-6">
                <Alert color={ACCENT2}>
                  <strong style={{ color: ACCENT2 }}>BB Docking pathway → MTS:</strong><br />
                  CPLANE1 LOF → CPLANE complex partially disrupted →
                  BB migration to apical membrane impaired → BB misdocked (wrong angle) →
                  cilia short (50–70% WT) and misoriented →
                  IFT-A/B entry geometry aberrant → reduced Hh gradient at cerebellar progenitors →
                  <strong> CEREBELLAR VERMIS HYPOPLASIA → MOLAR TOOTH SIGN (100%)</strong>
                </Alert>
              </div>
              <div className="col-md-6">
                <Alert color={ACCENT6}>
                  <strong style={{ color: ACCENT6 }}>PCP pathway → Polydactyly (~24%):</strong><br />
                  CPLANE1 LOF → PCP axis not transmitted to BB positioning in limb bud mesenchyme →
                  cilia geometry misaligned → GLI3 full-length / repressor ratio aberrant →
                  posterior digit specification expanded →
                  <strong> POSTAXIAL POLYDACTYLY (~24%; higher than average JBTS)</strong>
                </Alert>
              </div>
            </div>
          </Section>

          {/* DDx pearls */}
          <Section title="DDx Pearls" color={ACCENT5}>
            <ul className="small mb-0">
              {(overview.ddx_pearls || []).map((p, i) => <li key={i} className="mb-1">{p}</li>)}
            </ul>
          </Section>

          {/* Ethnic breakdown */}
          <Section title={`Ethnic Distribution (n=${N_COHORT}, seed ${SEED})`} color={ACCENT}>
            <DataTable
              accent={ACCENT}
              headers={['Ethnicity', 'n', '%']}
              rows={Object.entries(overview.ethnic_breakdown || {}).map(([eth, n]) =>
                [eth, n, `${Math.round(n / N_COHORT * 100)}%`])}
            />
          </Section>
        </div>
      )}

      {/* ── Tab 1: Diagnostic Breakdown ── */}
      {tab === 1 && breakdown && (
        <div>
          {/* Phenotype prevalence bar */}
          <Section title="Phenotype Prevalence (% of cohort)" color={ACCENT}>
            <div className="row">
              {Object.entries(breakdown.phenotype_prevalence || {}).map(([k, v]) => {
                const label = k.replace(/_/g, ' ').replace(/pct$/, '').replace(/100pct_diagnostic$/, '(100%)').replace(/100pct\)$/, '(100%)');
                return (
                  <div key={k} className="col-12 col-md-6 mb-2">
                    <div className="d-flex align-items-center gap-2">
                      <span className="small text-muted" style={{ minWidth: 200 }}>{label}</span>
                      <div className="flex-grow-1 bg-light rounded" style={{ height: 16 }}>
                        <div className="rounded h-100" style={{ width: `${v}%`, background: ACCENT + '99' }} />
                      </div>
                      <span className="small fw-bold" style={{ minWidth: 36, color: ACCENT }}>{v}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </Section>

          {/* Cilia length distribution */}
          <Section title="Cilia Length Distribution (BB misdocking → shortened cilia)" color={ACCENT8}>
            <DataTable
              accent={ACCENT8}
              headers={['Cilia Length Category', 'n', '%']}
              rows={Object.entries(breakdown.cilia_length_distribution || {}).map(([k, v]) =>
                [k, v, `${Math.round(v / N_COHORT * 100)}%`])}
            />
          </Section>

          {/* BB docking angle distribution */}
          <Section title="Basal Body Docking Angle Distribution" color={ACCENT3}>
            <DataTable
              accent={ACCENT3}
              headers={['BB Docking Category', 'n', '%']}
              rows={Object.entries(breakdown.bb_docking_angle_distribution || {}).map(([k, v]) =>
                [k, v, `${Math.round(v / N_COHORT * 100)}%`])}
            />
          </Section>

          {/* Allele classes */}
          <Section title="Allele Class Distribution" color={ACCENT2}>
            <DataTable
              accent={ACCENT2}
              headers={['Allele Class', 'n', '%']}
              rows={Object.entries(breakdown.allele_class_distribution || {}).map(([k, v]) =>
                [k, v, `${Math.round(v / N_COHORT * 100)}%`])}
            />
          </Section>

          {/* Polydactyly PCP analysis */}
          <Section title="Polydactyly — PCP Mechanism Analysis" color={ACCENT6}>
            <DataTable
              accent={ACCENT6}
              headers={['Category', 'n', '%']}
              rows={Object.entries(breakdown.polydactyly_pcp_analysis || {}).map(([k, v]) =>
                [k, v, `${Math.round(v / N_COHORT * 100)}%`])}
            />
          </Section>

          {/* Key variants */}
          <Section title="Key CPLANE1 Variants in JBTS33" color={ACCENT}>
            <DataTable
              accent={ACCENT}
              headers={['Variant', 'Domain', 'Population', 'Severity']}
              rows={(breakdown.key_variants || []).map(v => [
                v.variant, v.domain, v.population, v.severity
              ])}
            />
          </Section>

          {/* Cohort table */}
          <Section title={`Cohort Table (first 20 of ${N_COHORT})`} color={ACCENT5}>
            <DataTable
              accent={ACCENT5}
              headers={['ID', 'Ethnicity', 'MTS', 'OFC', 'Ataxia', 'Hypotonia', 'ID Severity', 'Retinal', 'Polydactyly', 'Variant 1']}
              rows={(breakdown.cohort_table || []).slice(0, 20).map(p => [
                p.id, p.ethnicity, p.mts, p.ofc, p.ataxia, p.hypotonia, p.id_severity, p.retinal, p.polydactyly, p.variant_1
              ])}
            />
          </Section>
        </div>
      )}

      {/* ── Tab 2: CPLANE Complex Pearls ── */}
      {tab === 2 && (
        <div>
          <Section title="CPLANE1 — Clinical Expert Pearls" color={ACCENT}>
            <DataTable
              accent={ACCENT}
              headers={['Topic', 'Pearl']}
              rows={[
                ['CPLANE complex', 'CPLANE1 + INTURNED (INTU) + FUZZY (FUZ) form an obligate cytoplasmic complex. CPLANE1 scaffolds INTU (PCP signal transducer) and FUZ (vesicular trafficking effector). Disruption of any member impairs BB apical docking.'],
                ['BB docking vs TZ gate', 'CPLANE1 acts upstream of the TZ — it governs whether the BB reaches and docks at the apical membrane. The TZ itself (B9 complex, MKS proteins) is intact in JBTS33. This is why no MKS phenotype occurs: the TZ gate is competent, just mispositioned.'],
                ['Normal OFC — key discriminator', 'JBTS33 OFC is normal at birth and throughout life. Microcephaly is not a feature. Any Joubert presentation with OFC ≤ −2 SD → prioritise KIF14 (JBTS32) instead.'],
                ['GT335 IF — normal', 'Axonemal polyglutamylation (GT335 signal) is normal in JBTS33 cilia (contrast JBTS29/TOGARAM1 where GT335 is pathognomonic reduced). GT335 intact + short cilia + MTS → consider JBTS33/CPLANE1.'],
                ['ARL13B IF — present', 'ARL13B is present in JBTS33 cilia (TZ intact). Absent ARL13B → TZ or IFT-A cargo-adaptor defect (JBTS30/TULP3). The combination of ARL13B present + shortened cilia + normal GT335 + MTS + polydactyly ~24% is strongly suggestive of CPLANE1.'],
                ['Polydactyly — PCP enriched', 'Postaxial polydactyly ~24% in JBTS33 — higher than the ~18% Joubert average. Mechanism: PCP-driven BB misalignment in limb bud → aberrant GLI3R/FL ratio → posterior digit formation. Co-assess VANGL2, CELSR1, PRICKLE1 in polydactyly-prominent families.'],
                ['INTU co-sequencing', 'INTURNED (INTU) is an obligate CPLANE1 partner. In families with one CPLANE1 pathogenic allele and unexplained second hit, co-sequence INTU. FUZ (Fuzzy) is the third partner and should be assessed in orofaciodigital + ciliopathy overlap.'],
                ['Renal surveillance', 'NPHP-like renal disease in ~18%. Annual urine ACR + eGFR starting at diagnosis. Renal ultrasound at age 5, then every 3 years. Early NPHP detection prevents progression to ESRD.'],
                ['Retinal dystrophy', '~22% rod-cone dystrophy. Electroretinogram (ERG) at diagnosis, repeat at 5 years. Progressive; low-vision aids if >40% visual field loss. No CPLANE1-specific retinal treatment beyond standard care.'],
                ['No MKS tier', 'All published JBTS33 patients are liveborn — biallelic CPLANE1 LOF is not equivalent to MKS-lethal phenotype. Prenatal: MTS on fetal MRI ≥ 20 weeks with normal head circumference and polydactyly → CPLANE1 high on list.'],
              ]}
            />
          </Section>

          <Section title="CPLANE1 BB-Docking Cascade" color={ACCENT3}>
            <Alert color={ACCENT3}>
              <strong>PCP signal → CPLANE complex → BB docking → ciliogenesis:</strong><br />
              VANGL2 asymmetric distribution (PCP axis) →
              INTU reads PCP gradient → recruits CPLANE1 scaffold →
              CPLANE1 engages FUZ → FUZ activates RAB11/EHD vesicular route →
              BB-directed vesicles transport appendage proteins + membrane phosphoinositides →
              BB migrates apically and docks at correct angle →
              IFT-A/B machinery correctly positioned →
              cilia grow (full length, directional orientation) →
              Hedgehog gradient sensed correctly by cerebellar progenitors →
              normal vermis development (no MTS).
              <br /><br />
              <strong>CPLANE1 LOF:</strong> FUZ engagement impaired →
              BB mis-docked (angle off-axis, 15–30°+) → IFT entry geometry aberrant →
              cilia shortened (50–70% WT) → Hh gradient reduced → vermis hypoplasia → MTS.
            </Alert>
          </Section>

          <Section title="Comparison with Adjacent JBTS Subtypes" color={ACCENT5}>
            <DataTable
              accent={ACCENT5}
              headers={['Feature', 'JBTS32 KIF14', 'JBTS33 CPLANE1', 'JBTS34 B9D2']}
              rows={[
                ['Gene role', 'Kinesin motor / cytokinesis + cilia length', 'PCP effector / BB docking', 'TZ B9-complex / diffusion barrier'],
                ['MTS', 'Yes (100%)', 'Yes (100%)', 'Yes (100%)'],
                ['Microcephaly', 'Yes (100%, OFC ≤ −2 SD)', 'No (normal OFC)', 'No (normal OFC)'],
                ['MKS tier', 'No', 'No', 'Yes (null → MKS10 lethal)'],
                ['Polydactyly', 'No (0%)', '~24% (PCP-enriched)', '~20% (Hh)'],
                ['Cilia length', 'Dysregulated (longer/aberrant)', '50–70% WT (short, misdocked)', 'Variable (TZ gate impaired)'],
                ['GT335 signal', 'Normal', 'Normal', 'Normal'],
                ['ARL13B IF', 'Present', 'Present', 'Reduced/absent (TZ leaky)'],
                ['Hepatic CHF', 'Rare (<5%)', 'Mild ~10%', 'Yes ~18% (TZ biliary)'],
                ['Allelic disease', 'MCPH20 (null)', 'None established', 'MKS10 (null)'],
              ]}
            />
          </Section>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && defs && (
        <div>
          <Section title="Definitions — CPLANE1 / JBTS33 / PCP Ciliogenesis Biology" color={ACCENT}>
            {(defs.definitions || []).map((d, i) => (
              <div key={i} className="mb-3 p-3 rounded" style={{ background: ACCENT + '08', borderLeft: `3px solid ${ACCENT}` }}>
                <div className="fw-bold small mb-1" style={{ color: ACCENT }}>{d.term}</div>
                <div className="small text-muted">{d.definition}</div>
              </div>
            ))}
          </Section>
        </div>
      )}

      {/* Footer nav */}
      <div className="mt-4 pt-3 border-top d-flex justify-content-between align-items-center">
        <Link href="/jbts32" className="btn btn-sm btn-outline-secondary">← JBTS32 KIF14</Link>
        <Link href="/" className="btn btn-sm btn-outline-primary">⌂ Home</Link>
        <Link href="/jbts34" className="btn btn-sm btn-outline-secondary">JBTS34 B9D2 →</Link>
      </div>
    </div>
  );
}
