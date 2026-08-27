'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'NPHP1 TZ Y-Link & Allele Switch', 'Definitions'];

// JBTS4 colour scheme — NPHP1 / Nephrocystin-1 / renal-neurological / large deletion
const ACCENT  = '#1a237e';   // deep indigo — JBTS/MTS neurological
const ACCENT2 = '#b71c1c';   // deep red — renal NPHP-type / high renal burden
const ACCENT3 = '#006064';   // dark cyan — transition zone molecular mechanism
const ACCENT4 = '#1b5e20';   // deep green — MLPA diagnostic pearl / deletion detection
const ACCENT5 = '#e65100';   // burnt orange — oculomotor apraxia / neonatal features
const ACCENT6 = '#4a148c';   // deep purple — AHI1 partner / NPHP1 complex
const ACCENT7 = '#f57f17';   // amber — allele-class switch / phenotype rule
const ACCENT8 = '#37474f';   // dark slate — cerebellar ataxia / movement
const ACCENT9 = '#880e4f';   // deep crimson — retinal dystrophy

const SEED = 415;
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

function AlertBox({ color, title, children }) {
  return (
    <div className="alert mb-3" style={{ borderLeft: `5px solid ${color}`, background: '#fafafa' }}>
      <strong style={{ color }}>{title}</strong>
      <div className="mt-1 small">{children}</div>
    </div>
  );
}

function Section({ title, color, children }) {
  return (
    <div className="mb-4">
      <h6 className="fw-bold mb-2" style={{ color }}>{title}</h6>
      {children}
    </div>
  );
}

function DataTable({ headers, rows }) {
  return (
    <div className="table-responsive mb-3">
      <table className="table table-sm table-bordered table-hover small mb-0">
        <thead className="table-dark">
          <tr>{headers.map((h, i) => <th key={i}>{h}</th>)}</tr>
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

// ── Tab: Overview ────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { kpis = [], hallmark, critical_diagnostic_pearl, allele_phenotype_rule, prevalence, first_description } = data;
  return (
    <div>
      <AlertBox color={ACCENT2} title="⚠ CRITICAL DIAGNOSTIC PEARL — MLPA MANDATORY">
        {critical_diagnostic_pearl}
      </AlertBox>
      <AlertBox color={ACCENT7} title="⚡ ALLELE-CLASS → PHENOTYPE SWITCH RULE">
        {allele_phenotype_rule}
      </AlertBox>
      <AlertBox color={ACCENT} title="🧬 JBTS4 Hallmark">
        {hallmark}
      </AlertBox>

      <div className="row g-2 mb-4">
        {kpis.map((k, i) => (
          <KPI key={i} label={k.label} value={k.value}
            color={[ACCENT,ACCENT2,ACCENT9,ACCENT5,ACCENT8,ACCENT3,ACCENT5,ACCENT8,ACCENT4][i % 9]} />
        ))}
      </div>

      <div className="row g-3">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
              JBTS4 at a Glance
            </div>
            <div className="card-body small">
              <table className="table table-sm mb-0">
                <tbody>
                  <tr><td className="fw-bold">Gene</td><td>NPHP1 (Nephrocystin-1)</td></tr>
                  <tr><td className="fw-bold">OMIM Gene</td><td>#607100</td></tr>
                  <tr><td className="fw-bold">OMIM Disease</td><td>#609583 (JBTS4)</td></tr>
                  <tr><td className="fw-bold">Chromosome</td><td>2q13</td></tr>
                  <tr><td className="fw-bold">Protein</td><td>732 aa — TZ Y-link scaffold</td></tr>
                  <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive (biallelic LOF)</td></tr>
                  <tr><td className="fw-bold">JBTS Frequency</td><td>~1–3% of all Joubert syndrome</td></tr>
                  <tr><td className="fw-bold">Prevalence</td><td>{prevalence}</td></tr>
                  <tr><td className="fw-bold">First described</td><td>{first_description}</td></tr>
                  <tr><td className="fw-bold">Cohort N</td><td>{N_COHORT} patients (seed {SEED})</td></tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold" style={{ background: ACCENT2, color: '#fff' }}>
              NPHP1 Dual Phenotype — Same Gene, Opposite Brain Phenotype
            </div>
            <div className="card-body small">
              <DataTable
                headers={['Allele Class', 'Phenotype', 'MTS?', 'Renal?']}
                rows={[
                  ['Biallelic NULL (2× truncating / deletion)', 'NPHP1 (#256100) — Nephronophthisis', '❌ Absent', '✅ ESRD dominant'],
                  ['One NULL + Hypomorphic (e.g. Thr323Met)', 'JBTS4 (#609583) — MTS + renal', '✅ Present', '✅ NPHP-type ~45%'],
                  ['Two HYPOMORPHIC (rare)', 'JBTS4 mild form', '✅ Present', '⚠ Mild/absent'],
                ]}
              />
              <div className="alert alert-warning small mb-0 py-2">
                <strong>Critical:</strong> Biallelic null → NO Molar Tooth Sign. NPHP1 allele class predicts brain vs renal phenotype completely.
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="row g-3 mt-1">
        <div className="col-md-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: ACCENT4, color: '#fff' }}>
              🔬 MLPA Diagnostic Pearl — Large ~610 kb NPHP1 Deletion
            </div>
            <div className="card-body small">
              <div className="row">
                <div className="col-md-8">
                  <ul className="mb-0">
                    <li><strong>~50–60% of all NPHP1 disease alleles</strong> are this large genomic deletion</li>
                    <li>Located in inverted repeat region at Chr 2q13 — prone to non-allelic homologous recombination</li>
                    <li><strong>INVISIBLE to WES, targeted panels, and Sanger sequencing</strong> — requires MLPA, aCGH, or long-read WGS</li>
                    <li>Homozygous deletion → NPHP1 (pure nephronophthisis, NO MTS)</li>
                    <li>Heterozygous deletion + hypomorphic missense → JBTS4 (MTS present)</li>
                    <li><strong>Rule: Never close NPHP1 workup without MLPA if WES is negative</strong></li>
                  </ul>
                </div>
                <div className="col-md-4">
                  <div className="border rounded p-2 text-center" style={{ background: '#e8f5e9' }}>
                    <div className="fw-bold" style={{ color: ACCENT4 }}>Detection Rate by Method</div>
                    <div className="mt-2">
                      <div>WES alone: <strong style={{ color: ACCENT2 }}>MISSES ~50%</strong></div>
                      <div>MLPA: <strong style={{ color: ACCENT4 }}>Detects deletion ✓</strong></div>
                      <div>aCGH: <strong style={{ color: ACCENT4 }}>Detects deletion ✓</strong></div>
                      <div>Long-read WGS: <strong style={{ color: ACCENT4 }}>Detects deletion ✓</strong></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Tab: Diagnostic Breakdown ─────────────────────────────────────────────────
function BreakdownTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { allele_class_summary = [], ethnicity_distribution = [], age_at_diagnosis_distribution = [],
          mts_distribution = [], feature_cooccurrence = [], mlpa_impact = {}, renal_severity_detail = {} } = data;
  return (
    <div>
      <div className="row g-3">
        <div className="col-md-6">
          <Section title="Allele-Class Distribution" color={ACCENT7}>
            <DataTable
              headers={['Allele Class', 'N', '%']}
              rows={allele_class_summary.map(r => [r.class, r.n, `${r.pct}%`])}
            />
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Ethnicity Distribution" color={ACCENT}>
            <DataTable
              headers={['Ethnicity', 'N', '%']}
              rows={ethnicity_distribution.map(r => [r.ethnicity, r.n, `${r.pct}%`])}
            />
          </Section>
        </div>
      </div>

      <div className="row g-3">
        <div className="col-md-6">
          <Section title="Age at Diagnosis" color={ACCENT8}>
            <DataTable
              headers={['Age Group', 'N', '%']}
              rows={age_at_diagnosis_distribution.map(r => [r.bin, r.n, `${r.pct}%`])}
            />
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Molar Tooth Sign (MTS) Distribution" color={ACCENT}>
            <DataTable
              headers={['Category', 'N', '%']}
              rows={mts_distribution.map(r => [r.label, r.n, `${r.pct}%`])}
            />
          </Section>
        </div>
      </div>

      <div className="row g-3">
        <div className="col-md-6">
          <Section title="Feature Co-occurrence" color={ACCENT3}>
            <DataTable
              headers={['Feature Pair', 'N', '%']}
              rows={feature_cooccurrence.map(r => [r.pair, r.n, `${r.pct}%`])}
            />
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="MLPA Diagnostic Impact" color={ACCENT4}>
            <div className="card border-success mb-2">
              <div className="card-body small">
                <div><strong>Missed by WES alone:</strong> {mlpa_impact.missed_by_wes_alone} patients ({mlpa_impact.pct_missed}%)</div>
                <div className="mt-2 text-muted">{mlpa_impact.note}</div>
              </div>
            </div>
          </Section>
          <Section title="Renal NPHP-type Severity" color={ACCENT2}>
            <div className="card border-danger mb-2">
              <div className="card-body small">
                <div><strong>NPHP present:</strong> {renal_severity_detail.nphp_present} patients ({renal_severity_detail.nphp_pct}%)</div>
                <div className="mt-2 text-muted">{renal_severity_detail.note}</div>
              </div>
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}

// ── Tab: NPHP1 TZ & Allele Switch ───────────────────────────────────────────
function MechanismTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { gene_card = {}, key_variants = [], nphp1_module_comparison = [] } = data;
  return (
    <div>
      <div className="row g-3 mb-3">
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold" style={{ background: ACCENT3, color: '#fff' }}>
              NPHP1 Protein Domains (732 aa) — TZ Y-Link Scaffold
            </div>
            <div className="card-body small">
              {(gene_card.domains || []).map((d, i) => (
                <div key={i} className="mb-2 p-2 rounded" style={{ background: '#f5f5f5', borderLeft: `4px solid ${[ACCENT,ACCENT3,ACCENT6,ACCENT4][i%4]}` }}>
                  <div className="fw-bold">{d.name}</div>
                  <div className="text-muted">{d.role}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card shadow-sm h-100">
            <div className="card-header fw-bold" style={{ background: ACCENT7, color: '#fff' }}>
              ⚡ Allele-Class → Phenotype Switch (Unique to NPHP1)
            </div>
            <div className="card-body small">
              <div className="p-2 mb-2 rounded" style={{ background: '#fff3e0', border: `1px solid ${ACCENT7}` }}>
                <strong>Mechanism of disease:</strong>
                <div className="mt-1 text-muted">{gene_card.mechanism_of_disease}</div>
              </div>
              <div className="p-2 rounded" style={{ background: '#e8eaf6', border: `1px solid ${ACCENT}` }}>
                <strong>Allele-phenotype switch:</strong>
                <div className="mt-1 text-muted">{gene_card.allele_phenotype_switch}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <Section title="Key Pathogenic Variants" color={ACCENT7}>
        <DataTable
          headers={['Variant', 'Domain', 'Consequence', 'Ethnicity']}
          rows={key_variants.map(v => [v.variant, v.domain, v.consequence, v.ethnicity])}
        />
      </Section>

      <Section title="NPHP1 Module — JBTS Gene Comparison" color={ACCENT3}>
        <DataTable
          headers={['Gene', 'JBTS', 'Chr', 'Module', 'Allele Switch', 'Renal', 'Retinal', 'OMA']}
          rows={nphp1_module_comparison.map(r => [r.gene, r.jbts, r.chr, r.module, r.allele_switch, r.renal, r.retinal, r.oma])}
        />
      </Section>
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { gene_card = {}, treatment_summary = [], ddx_table = [] } = data;
  return (
    <div>
      <div className="row g-3 mb-3">
        <div className="col-md-12">
          <div className="card shadow-sm">
            <div className="card-header fw-bold" style={{ background: ACCENT, color: '#fff' }}>
              NPHP1 Gene Card — Nephrocystin-1
            </div>
            <div className="card-body small">
              <div className="row">
                <div className="col-md-3">
                  <div className="mb-2"><strong>Gene:</strong> {gene_card.gene}</div>
                  <div className="mb-2"><strong>Full name:</strong> {gene_card.full_name}</div>
                  <div className="mb-2"><strong>OMIM:</strong> #{gene_card.omim}</div>
                  <div className="mb-2"><strong>Chr:</strong> {gene_card.chromosome}</div>
                  <div className="mb-2"><strong>Protein:</strong> {gene_card.protein_size}</div>
                </div>
                <div className="col-md-9">
                  <div className="mb-2"><strong>Function:</strong> {gene_card.function}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <Section title="Treatment & Management" color={ACCENT3}>
        <div className="card shadow-sm">
          <div className="card-body small">
            <ol className="mb-0">
              {treatment_summary.map((t, i) => <li key={i} className="mb-1">{t}</li>)}
            </ol>
          </div>
        </div>
      </Section>

      <Section title="Differential Diagnosis" color={ACCENT8}>
        <DataTable
          headers={['Disease', 'Key Distinguishing Feature']}
          rows={ddx_table.map(r => [r.disease, r.key_difference])}
        />
      </Section>
    </div>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function JBTS4Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]     = useState(null);
  const [breakdown, setBreakdown]   = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts4/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts4/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts4/definitions`).then(r => r.json()),
    ]).then(([ov, br, def]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefinitions(def);
    }).catch(e => setError(String(e)));
  }, []);

  return (
    <div className="container-fluid py-3">
      <div className="d-flex align-items-center mb-1">
        <Link href="/" className="btn btn-sm btn-outline-secondary me-2">← Home</Link>
        <h4 className="mb-0" style={{ color: ACCENT }}>
          🧬 JBTS4 — Joubert Syndrome Type 4
        </h4>
      </div>
      <div className="text-muted small mb-3">
        <strong>NPHP1 (Nephrocystin-1)</strong> · 2q13 · TZ Y-Link Scaffold · OMIM #609583 ·
        Dual phenotype: NPHP1 (biallelic null) ↔ JBTS4 (null + hypomorphic) ·
        40-patient cohort (seed {SEED})
      </div>

      {error && <div className="alert alert-danger small">{error}</div>}

      <ul className="nav nav-tabs mb-3">
        {TABS.map((t, i) => (
          <li key={i} className="nav-item">
            <button className={`nav-link${tab === i ? ' active' : ''}`} onClick={() => setTab(i)}>{t}</button>
          </li>
        ))}
      </ul>

      {tab === 0 && <OverviewTab     data={overview} />}
      {tab === 1 && <BreakdownTab    data={breakdown} />}
      {tab === 2 && <MechanismTab    data={definitions} />}
      {tab === 3 && <DefinitionsTab  data={definitions} />}
    </div>
  );
}
