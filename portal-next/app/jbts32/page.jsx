'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Kinesin Motor Pearls', 'Definitions'];

// JBTS32 colour scheme — KIF14 / Kinesin Motor / Microcephaly / Cytokinesis
// Deep indigo for kinesin motor identity; red for microcephaly alert; teal for cytokinesis
const ACCENT   = '#283593';   // deep indigo — KIF14 kinesin motor identity
const ACCENT2  = '#b71c1c';   // crimson — microcephaly (distinctive JBTS32 feature)
const ACCENT3  = '#00695c';   // dark teal — cytokinesis / midbody ring
const ACCENT4  = '#2e7d32';   // forest green — no MKS tier / all liveborn
const ACCENT5  = '#37474f';   // slate — domain matrix / tables
const ACCENT6  = '#e65100';   // deep orange — MCPH20 allelic / allele severity alert
const ACCENT7  = '#4527a0';   // purple — cortical malformation / microcephaly depth
const ACCENT8  = '#1565c0';   // royal blue — cilia length regulation
const ACCENT9  = '#1b5e20';   // dark green — cerebellar ataxia / MTS
const ACCENT10 = '#795548';   // brown — intellectual disability

const SEED = 483;
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

export default function JBTS32Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);
  const [loading, setLoading]     = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/jbts32/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts32/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts32/definitions`).then(r => r.json()),
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
              JBTS32 — KIF14 Joubert Syndrome Type 32
            </h4>
            <div className="text-muted small">
              KIF14 Kinesin Motor · Microcephaly + MTS · Cytokinesis + Cilia Length Regulation ·
              1q31.3 · OMIM <a href="#" style={{ color: ACCENT2 }}>#616651</a> ·
              Allelic: MCPH20 (<a href="#" style={{ color: ACCENT6 }}>#617913</a>) ·
              {N_COHORT}-patient cohort · seed {SEED}
            </div>
          </div>
        </div>
      </div>

      {/* Microcephaly alert — defining JBTS32 feature */}
      <Alert color={ACCENT2}>
        <strong style={{ color: ACCENT2 }}>⚠ MICROCEPHALY IN JBTS32 — THE DISTINCTIVE JOUBERT FEATURE:</strong>&nbsp;
        JBTS32 (KIF14) is one of the very few Joubert syndrome subtypes with consistent, clinically significant
        PRIMARY MICROCEPHALY (OFC ≤ −2 SD at birth) in 100% of patients. Most other JBTS subtypes have normal OFC.
        Mechanism: partial cytokinesis failure in neuroprogenitors → reduced cortical neuron output → small brain.
        Detection of microcephaly in a Joubert patient → prioritise KIF14 on the gene panel.
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
            <KPI label="MTS (all JBTS32)" value="100%" color={ACCENT9} />
            <KPI label="Primary Microcephaly" value="100%" color={ACCENT2} />
            <KPI label="Cerebellar Ataxia" value={kpis.cerebellar_ataxia_pct} color={ACCENT9} />
            <KPI label="Neonatal Hypotonia" value={kpis.neonatal_hypotonia_pct} color={ACCENT3} />
            <KPI label="Intellectual Disability" value={kpis.intellectual_disability} color={ACCENT10} />
            <KPI label="Oculomotor Apraxia" value={kpis.oculomotor_apraxia_pct} color={ACCENT8} />
          </div>
          <div className="row mb-3">
            <KPI label="Breathing Dysregulation" value={kpis.breathing_dysreg_pct} color={ACCENT3} />
            <KPI label="Retinal Rod-Cone" value={kpis.retinal_pct} color={ACCENT8} />
            <KPI label="Renal (mild)" value={kpis.renal_pct} color={ACCENT7} />
            <KPI label="Polydactyly" value="0% (none)" color={ACCENT4} />
            <KPI label="Thorax Normal" value="100%" color={ACCENT4} />
            <KPI label="All Liveborn" value="100%" color={ACCENT4} />
          </div>

          {/* Gene card */}
          <Section title="Gene: KIF14 (*611498)" color={ACCENT}>
            <div className="row">
              <div className="col-md-6">
                <DataTable
                  accent={ACCENT}
                  headers={['Property', 'Value']}
                  rows={[
                    ['Gene', 'KIF14 (Kinesin Family Member 14)'],
                    ['OMIM Gene', '*611498'],
                    ['OMIM Disease', '#616651 (JBTS32)'],
                    ['OMIM Allelic', '#617913 (MCPH20 — biallelic null)'],
                    ['Chromosome', '1q31.3'],
                    ['Protein length', '~1,648 aa'],
                    ['Protein class', 'Plus-end-directed kinesin motor'],
                    ['Inheritance', 'Autosomal Recessive — biallelic hypomorphic LOF'],
                  ]}
                />
              </div>
              <div className="col-md-6">
                <DataTable
                  accent={ACCENT3}
                  headers={['Domain', 'Residues', 'Key Function']}
                  rows={[
                    ['FHA domain', 'aa 1–100', 'Phosphothreonine binding; kinetochore / spindle localisation; CIT binding'],
                    ['Motor domain', 'aa 101–700', 'ATPase; microtubule binding; plus-end processive stepping; Switch-I/II loops'],
                    ['Neck CC / Stalk I', 'aa 701–900', 'Homodimerisation; RACGAP1 binding; motor processivity'],
                    ['Extended stalk / Tail', 'aa 901–1,648', 'Cargo binding; PRC1 contact; microtubule bundling; cilia length regulation'],
                  ]}
                />
              </div>
            </div>
          </Section>

          {/* JBTS32 vs MCPH20 */}
          <Section title="⚠ JBTS32 vs MCPH20 (same gene KIF14 — allele-class distinction)" color={ACCENT6}>
            <Alert color={ACCENT6}>
              <strong>Biallelic null/truncating → MCPH20</strong> (#617913): severe primary microcephaly (OFC ≤ −5 SD),
              severe intellectual disability, MTS may be absent/subtle, cytokinesis catastrophically impaired.&nbsp;
              <strong>Biallelic hypomorphic missense → JBTS32</strong> (#616651): moderate microcephaly (OFC −2 to −4 SD) +
              MTS + Joubert cerebellar features; residual KIF14 activity (~40–60% WT) prevents MCPH20-level cytokinesis failure.
              <br /><strong>Clinical discriminator:</strong> Brain MRI — MTS present → JBTS32; MTS absent with severe microcephaly → MCPH20 allele class.
              OFC: ≥ −4 SD with MTS → JBTS32; ≤ −5 SD without MTS → MCPH20.
            </Alert>
          </Section>

          {/* Cytokinesis + Ciliogenesis pathway */}
          <Section title="KIF14 Dual Pathogenic Pathway" color={ACCENT3}>
            <div className="row">
              <div className="col-md-6">
                <Alert color={ACCENT3}>
                  <strong style={{ color: ACCENT3 }}>Cytokinesis pathway → Microcephaly:</strong><br />
                  KIF14 hypomorphic → partial cytokinesis failure in cortical neuroprogenitors →
                  midbody ring incomplete → cleavage furrow regression → binucleate cells →
                  mitotic checkpoint → apoptosis → reduced cortical neuron output →
                  <strong> MODERATE MICROCEPHALY (OFC −2 to −4 SD)</strong>
                </Alert>
              </div>
              <div className="col-md-6">
                <Alert color={ACCENT8}>
                  <strong style={{ color: ACCENT8 }}>Ciliogenesis pathway → MTS:</strong><br />
                  KIF14 hypomorphic → aberrant cilia length regulation at cerebellar progenitors →
                  dysregulated GLI2/GLI3 Hedgehog gradient → granule cell proliferation impaired →
                  cerebellar vermis hypoplasia →
                  <strong> MOLAR TOOTH SIGN on MRI (100%)</strong>
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
                const label = k.replace(/_/g, ' ').replace(/pct$/, '').replace(/100pct$/, '(100%)');
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

          {/* Microcephaly OFC distribution */}
          <Section title="Microcephaly OFC Distribution" color={ACCENT2}>
            <DataTable
              accent={ACCENT2}
              headers={['OFC Category', 'n', '%']}
              rows={Object.entries(breakdown.microcephaly_ofc_distribution || {}).map(([k, v]) =>
                [k, v, `${Math.round(v / N_COHORT * 100)}%`])}
            />
          </Section>

          {/* MTS severity */}
          <Section title="MTS Severity Distribution" color={ACCENT9}>
            <DataTable
              accent={ACCENT9}
              headers={['MTS Category', 'n', '%']}
              rows={Object.entries(breakdown.mts_severity_distribution || {}).map(([k, v]) =>
                [k, v, `${Math.round(v / N_COHORT * 100)}%`])}
            />
          </Section>

          {/* Allele classes */}
          <Section title="Allele Class Distribution" color={ACCENT3}>
            <DataTable
              accent={ACCENT3}
              headers={['Allele Class', 'n', '%']}
              rows={Object.entries(breakdown.allele_class_distribution || {}).map(([k, v]) =>
                [k, v, `${Math.round(v / N_COHORT * 100)}%`])}
            />
          </Section>

          {/* Cortical malformation */}
          <Section title="Cortical Malformation (microcephaly-related)" color={ACCENT7}>
            <DataTable
              accent={ACCENT7}
              headers={['Cortical Finding', 'n', '%']}
              rows={Object.entries(breakdown.cortical_malformation_distribution || {}).map(([k, v]) =>
                [k, v, `${Math.round(v / N_COHORT * 100)}%`])}
            />
          </Section>

          {/* Key variants */}
          <Section title="Key KIF14 Variants in JBTS32" color={ACCENT}>
            <DataTable
              accent={ACCENT}
              headers={['Variant', 'Domain', 'Population', 'Severity']}
              rows={(breakdown.key_variants || []).map(v => [
                v.variant, v.domain, v.population, v.severity
              ])}
            />
          </Section>

          {/* Cohort table (first 20) */}
          <Section title={`Cohort Table (first 20 of ${N_COHORT})`} color={ACCENT5}>
            <DataTable
              accent={ACCENT5}
              headers={['ID', 'Ethnicity', 'MTS', 'Microcephaly OFC', 'Ataxia', 'ID Severity', 'Retinal', 'Variant 1']}
              rows={(breakdown.cohort_table || []).slice(0, 20).map(p => [
                p.id, p.ethnicity, p.mts, p.microcephaly, p.ataxia, p.id_severity, p.retinal, p.variant_1
              ])}
            />
          </Section>
        </div>
      )}

      {/* ── Tab 2: Kinesin Motor Pearls ── */}
      {tab === 2 && (
        <div>
          <Section title="KIF14 Kinesin Motor — Clinical Expert Pearls" color={ACCENT}>
            <DataTable
              accent={ACCENT}
              headers={['Topic', 'Pearl']}
              rows={[
                ['KIF14 Motor Class', 'Plus-end-directed kinesin; N-kinesin subfamily. Processively walks along antiparallel microtubule bundles at the spindle midzone during cytokinesis.'],
                ['Cytokinesis partner: RACGAP1', 'RACGAP1 (MgcRacGAP) is the principal effector: KIF14 translocates RACGAP1 to the midbody, where it activates RhoA for cleavage furrow ingression. RACGAP1 should be assessed in incomplete compound het KIF14 families.'],
                ['Cytokinesis partner: CIT kinase', 'Citron kinase (CIT/CITK) is recruited by KIF14 (via FHA domain phosphothreonine binding) to the midbody. CIT variants cause primary microcephaly (MCPH17). CIT co-assessment recommended in JBTS32 families.'],
                ['Cilia length: KIF14 as negative regulator', 'KIF14 depletion in vitro → elongated cilia. KIF14 hypomorphic → aberrant (likely elongated or dysregulated) cilia in vivo. This differs mechanistically from IFT-B defect subtypes (short cilia) — KIF14 cilia phenotype is length dysregulation, not simple shortening.'],
                ['FHA domain variants → MCPH20', 'Variants disrupting the FHA domain (aa 1–100) tend to cause MCPH20 (severe microcephaly, absent/subtle MTS) by abolishing CIT kinase binding and spindle localisation. Hypomorphic motor domain variants → JBTS32 (MTS + moderate microcephaly).'],
                ['Microcephaly OFC alert', 'OFC at birth: JBTS32 typically −2 to −4 SD. OFC ≤ −5 SD → suspect MCPH20 allele class. Sequential OFC measurements: JBTS32 microcephaly is postnatal-static (not progressive); progressive microcephaly suggests different diagnosis.'],
                ['Simplified gyral pattern', '~35% of JBTS32 patients have simplified gyral pattern or focal pachygyria on MRI. This reflects the combined cortical neuroprogenitor cytokinesis and cilia defect. Not seen in most other JBTS subtypes (discriminating MRI finding).'],
                ['Gene panel priority', 'Any Joubert presentation with microcephaly (OFC ≤ −2 SD) should trigger KIF14 prioritisation on the ciliopathy/Joubert gene panel. Standard Joubert panels (INPP5E, TMEM216, AHI1, CEP290, RPGRIP1L) will miss JBTS32.'],
                ['No MKS tier', 'KIF14 is a cytoplasmic kinesin motor, not a transition-zone (TZ) diffusion-barrier component. JBTS32 does not overlap with MKS/SRPS lethal perinatal forms. All JBTS32 patients are liveborn.'],
                ['Intellectual disability severity', 'ID in JBTS32 is generally moderate-to-severe (more severe than average JBTS) due to compound microcephaly + Joubert cerebellar deficit. Early neurodevelopmental intervention and occupational therapy are critical.'],
              ]}
            />
          </Section>

          <Section title="Cytokinesis Failure Cascade in JBTS32" color={ACCENT3}>
            <Alert color={ACCENT3}>
              <strong>KIF14 hypomorphic → partial cytokinesis failure in neuroprogenitors:</strong><br />
              Residual KIF14 activity (~40–60% WT) → RACGAP1 partially translocated to midbody →
              partial RhoA activation → incomplete cleavage furrow ingression (some cells regress) →
              binucleated neuroprogenitors → mitotic checkpoint activation (p53 pathway) →
              selective apoptosis of affected cells → overall neuroprogenitor pool reduced →
              fewer cortical neurons generated → MODERATE MICROCEPHALY (OFC −2 to −4 SD).
              <br /><br />
              Contrast MCPH20 (biallelic null): complete RACGAP1 translocation failure → nearly universal
              cytokinesis failure → massive neuroprogenitor apoptosis → SEVERE MICROCEPHALY (OFC ≤ −5 SD).
            </Alert>
          </Section>

          <Section title="Allelic Disease Comparison" color={ACCENT6}>
            <DataTable
              accent={ACCENT6}
              headers={['Feature', 'JBTS32 (hypomorphic)', 'MCPH20 (null/severe)']}
              rows={[
                ['OMIM', '#616651', '#617913'],
                ['MTS on MRI', 'Present (100%)', 'Absent or subtle'],
                ['Microcephaly OFC', '−2 to −4 SD (moderate)', '≤ −5 SD (severe)'],
                ['KIF14 residual function', '~40–60% WT', '<10% or zero'],
                ['Cytokinesis failure', 'Partial (moderate neuroprogenitor apoptosis)', 'Near-complete (massive apoptosis)'],
                ['Cortical gyral pattern', 'Simplified gyral ~35%', 'Markedly simplified / lissencephaly rare'],
                ['Intellectual disability', 'Moderate–severe', 'Severe–profound'],
                ['Cerebellar ataxia', 'Present (~80%)', 'Variable (often present)'],
                ['Allele types', 'Biallelic missense / splice (partial LOF)', 'Biallelic truncating / null'],
              ]}
            />
          </Section>
        </div>
      )}

      {/* ── Tab 3: Definitions ── */}
      {tab === 3 && defs && (
        <div>
          <Section title="Definitions — KIF14 / JBTS32 / Kinesin Motor Biology" color={ACCENT}>
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
        <Link href="/jbts31" className="btn btn-sm btn-outline-secondary">← JBTS31 CEP120</Link>
        <Link href="/" className="btn btn-sm btn-outline-primary">⌂ Home</Link>
        <Link href="/jbts34" className="btn btn-sm btn-outline-secondary">JBTS34 B9D2 →</Link>
      </div>
    </div>
  );
}
