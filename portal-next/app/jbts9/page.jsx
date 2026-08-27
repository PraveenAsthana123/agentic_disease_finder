'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'MKS6 Tier & COACH Pearl', 'Definitions'];

// JBTS9 colour scheme — CC2D2A / TZ scaffold / MKS6 / COACH / hepato-renal
const ACCENT  = '#880e4f';   // deep crimson — JBTS9 / MKS6 lethal-tier warning
const ACCENT2 = '#1a237e';   // deep indigo — TZ scaffold / MTS neurological
const ACCENT3 = '#004d40';   // deep teal — renal NPHP-like
const ACCENT4 = '#1b5e20';   // deep green — transplant / curative endpoint
const ACCENT5 = '#b71c1c';   // deep red — MKS6 lethal tier pearl
const ACCENT6 = '#e65100';   // burnt orange — hepatic fibrosis / COACH
const ACCENT7 = '#4a148c';   // deep purple — allele tier rule / coiled-coil
const ACCENT8 = '#37474f';   // dark slate — cerebellar ataxia / OMA
const ACCENT9 = '#bf360c';   // dark orange-red — polydactyly / JSOFD

const SEED = 425;
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
  const { kpis = [], hallmark, mks6_tier_pearl, coach_pearl, jsofd_pearl,
          first_description, gene_summary, phenotype_summary,
          allele_class_distribution = [] } = data;
  return (
    <div>
      <AlertBox color={ACCENT5} title="⚠ MKS6 LETHAL TIER — BIALLELIC NULL CC2D2A → MECKEL-GRUBER SYNDROME 6">
        {mks6_tier_pearl}
      </AlertBox>
      <AlertBox color={ACCENT6} title="⚠ COACH SYNDROME — CC2D2A IS THE SECOND MOST COMMON COACH GENE (after TMEM67)">
        {coach_pearl}
      </AlertBox>
      <AlertBox color={ACCENT9} title="⚠ JSOFD — POST-AXIAL POLYDACTYLY ~20% (KIF7/CC2D2A/TCTN SUBTYPE)">
        {jsofd_pearl}
      </AlertBox>

      <Section title="KPI Summary — JBTS9 Cohort (N=40, seed-425)" color={ACCENT2}>
        <div className="row g-2">
          {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
        </div>
      </Section>

      <div className="row">
        <div className="col-md-6">
          <Section title="Gene & Disease Summary" color={ACCENT2}>
            <table className="table table-sm table-bordered small">
              <tbody>
                <tr><td className="fw-bold">Gene</td><td>CC2D2A (OMIM *612013)</td></tr>
                <tr><td className="fw-bold">Disease JBTS9</td><td>Joubert Syndrome 9 (OMIM #612285)</td></tr>
                <tr><td className="fw-bold">Disease MKS6</td><td style={{color: ACCENT5, fontWeight:'bold'}}>Meckel-Gruber Syndrome 6 (OMIM #612284) — biallelic null → lethal</td></tr>
                <tr><td className="fw-bold">Chromosome</td><td>4p15.33</td></tr>
                <tr><td className="fw-bold">Protein</td><td>1620 aa — CC1 (TMEM67) / CC2 (RPGRIP1L) / C2 membrane / C-term CC (CEP290/B9D1)</td></tr>
                <tr><td className="fw-bold">Inheritance</td><td>Autosomal Recessive — biallelic LOF; allele class governs tier</td></tr>
                <tr><td className="fw-bold">Prevalence</td><td>~3–5% all JBTS; ~1/600,000–1,000,000 worldwide</td></tr>
                <tr><td className="fw-bold">MKS6 Tier</td><td style={{color: ACCENT5, fontWeight:'bold'}}>YES — biallelic null → MKS6 (perinatal lethal)</td></tr>
                <tr><td className="fw-bold">COACH Gene</td><td style={{color: ACCENT6}}>Yes — 2nd most common (after TMEM67); CHF ~25%</td></tr>
                <tr><td className="fw-bold">JSOFD (Polydactyly)</td><td>~20% post-axial polydactyly; JSOFD subtype</td></tr>
                <tr><td className="fw-bold">First Description</td><td>{first_description}</td></tr>
                <tr><td className="fw-bold">Hallmark</td><td>{hallmark}</td></tr>
              </tbody>
            </table>
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Phenotype Frequencies (N=40)" color={ACCENT8}>
            <DataTable
              headers={['Feature', 'Frequency', 'Notes']}
              rows={[
                ['Molar Tooth Sign (MTS)', '100%', 'Pathognomonic — all JBTS9'],
                ['Cerebellar Ataxia', `${phenotype_summary?.ataxia_pct ?? '–'}%`, 'Core feature; SARA tracking'],
                ['Neonatal Hypotonia', `${phenotype_summary?.hypotonia_pct ?? '–'}%`, 'Feeding difficulty in infancy'],
                ['Oculomotor Apraxia', `${phenotype_summary?.oma_pct ?? '–'}%`, 'Head thrust compensation'],
                ['Intellectual Disability', `${phenotype_summary?.id_pct ?? '–'}%`, 'Mild–moderate range'],
                ['Breathing Dysregulation', `${phenotype_summary?.breathing_pct ?? '–'}%`, 'Episodic apnea/hyperpnea'],
                ['Retinal Dystrophy', `${phenotype_summary?.retinal_pct ?? '–'}%', 'Rod-cone; ERG annual; coloboma ~15%'],
                ['Hepatic Fibrosis (CHF)', `${phenotype_summary?.hepatic_pct ?? '–'}%`, 'COACH subtype; portal HTN; varices risk'],
                ['Renal (NPHP-like TIN)', `${phenotype_summary?.renal_pct ?? '–'}%`, 'ESRD risk ~20yr median'],
                ['Polydactyly (JSOFD)', `${phenotype_summary?.polydactyly_pct ?? '–'}%`, 'Post-axial; skeletal survey mandatory'],
                ['Coloboma (COACH)', `${phenotype_summary?.coloboma_pct ?? '–'}%`, 'Retinochoroidal; COACH subtype'],
                ['MKS6 Lethal Tier', '100% if biallelic null', 'Encephalocele + polydactyly + PKD'],
              ]}
            />
          </Section>
        </div>
      </div>

      <Section title="Allele Class Distribution (Cohort)" color={ACCENT7}>
        <DataTable
          headers={['Allele Class', 'Count', '%']}
          rows={allele_class_distribution.map(a => [a.allele_class, a.count, `${a.pct}%`])}
        />
      </Section>

      <Section title="Gene Function Summary" color={ACCENT2}>
        <p className="small">{gene_summary}</p>
      </Section>
    </div>
  );
}

// ── Tab: Diagnostic Breakdown ────────────────────────────────────────────────
function BreakdownTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const {
    ethnicity_distribution = [],
    key_variants = [],
    domain_phenotype_matrix = [],
    pathway_steps = [],
    management = [],
    patient_table = [],
  } = data;

  return (
    <div>
      <Section title="Ethnicity Distribution" color={ACCENT2}>
        <DataTable
          headers={['Ethnicity', 'Count', '%']}
          rows={ethnicity_distribution.map(e => [e.ethnicity, e.count, `${e.pct}%`])}
        />
      </Section>

      <Section title="Key Pathogenic Variants (CC2D2A)" color={ACCENT5}>
        <DataTable
          headers={['Variant', 'Domain', 'Effect', 'Population', 'Phenotype', 'OMIM Class']}
          rows={key_variants.map(v => [
            <strong key={v.variant} style={{ color: ACCENT5 }}>{v.variant}</strong>,
            v.domain, v.effect, v.population, v.phenotype, v.omim_class,
          ])}
        />
      </Section>

      <Section title="Domain → Phenotype Severity Matrix" color={ACCENT7}>
        <DataTable
          headers={['Domain', 'Key Variants', 'Function Lost', 'Severity', 'Hepatic Risk', 'Renal Risk']}
          rows={domain_phenotype_matrix.map(d => [
            d.domain, d.key_variants, d.function_lost, d.severity, d.hepatic_risk, d.renal_risk,
          ])}
        />
      </Section>

      <Section title="CC2D2A → TZ Gate → Hedgehog / Hepatic / Renal Pathway" color={ACCENT2}>
        <DataTable
          headers={['Step', 'Normal Event', 'Effect When CC2D2A Lost']}
          rows={pathway_steps.map(s => [s.step, s.event, s.effect_when_lost])}
        />
      </Section>

      <Section title="Clinical Management" color={ACCENT4}>
        <DataTable
          headers={['Intervention', 'Timing', 'Rationale', 'Level']}
          rows={management.map(m => [m.intervention, m.timing, m.rationale, m.level])}
        />
      </Section>

      <Section title="Per-Patient Table (first 20)" color={ACCENT8}>
        <DataTable
          headers={['ID', 'Ethnicity', 'Allele', 'Age Dx', 'MTS', 'Ataxia', 'OMA', 'Retinal', 'Hepatic', 'Renal', 'Poly', 'ID', 'Breathing', 'MKS Tier']}
          rows={patient_table.map(p => [
            p.id, p.ethnicity, p.allele, p.age_dx_yr,
            p.mts, p.ataxia, p.oma, p.retinal, p.hepatic, p.renal,
            p.poly, p.id_, p.breathing, p.mks_tier,
          ])}
        />
      </Section>
    </div>
  );
}

// ── Tab: MKS6 Tier & COACH Pearl ────────────────────────────────────────────
function PearlTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <AlertBox color={ACCENT5} title="⚠ MKS6 ALLELE-TIER RULE — IDENTICAL TO RPGRIP1L (JBTS7/MKS5) AND TMEM67 (JBTS6/MKS3)">
        CC2D2A biallelic null variants → MKS6 (Meckel-Gruber Syndrome 6, OMIM #612284):
        perinatal lethal — occipital encephalocele + post-axial polydactyly + cystic kidneys.
        At least one hypomorphic missense allele required for JBTS9 (live birth). This tier
        rule is the SAME as RPGRIP1L (MKS5) and TMEM67 (MKS3). Parents each carrying one null
        CC2D2A allele have 25% risk of MKS6 lethal offspring. PGT-M strongly recommended
        for subsequent pregnancies with null-carrying parents.
      </AlertBox>

      <Section title="MKS Allele-Tier Comparison — Major JBTS/MKS Genes" color={ACCENT5}>
        <DataTable
          headers={['Gene', 'JBTS Type', 'Biallelic Null Tier', 'NPHP Tier', 'COACH?']}
          rows={[
            ['CC2D2A', 'JBTS9', 'MKS6 (perinatal lethal — encephalocele/polydactyly/PKD)', 'NPHP-like (rare biallelic missense)', 'Yes — 2nd most common COACH gene'],
            ['TMEM67', 'JBTS6', 'MKS3 (perinatal lethal)', 'NPHP11', 'Yes — MOST COMMON COACH gene (~80%)'],
            ['RPGRIP1L', 'JBTS7', 'MKS5 (perinatal lethal)', 'NPHP8', 'Mild CHF only (not full COACH)'],
            ['CEP290', 'JBTS5', 'MKS4 (perinatal lethal)', 'NPHP6', '—'],
            ['ARL13B', 'JBTS8', 'JBTS8 (live birth — NO MKS tier)', '—', '— (no CHF in ARL13B)'],
            ['AHI1', 'JBTS3', 'JBTS3 (live birth — no MKS tier)', '—', '—'],
          ]}
        />
      </Section>

      <Section title="COACH Syndrome — CC2D2A vs TMEM67 Comparison" color={ACCENT6}>
        <DataTable
          headers={['Feature', 'CC2D2A-COACH (JBTS9)', 'TMEM67-COACH (JBTS6)', 'Clinical Implication']}
          rows={[
            ['COACH frequency', '~15% of COACH cases', '~80% of COACH cases', 'Test TMEM67 first; CC2D2A second'],
            ['CHF severity', 'Mild–moderate; portal HTN less severe', 'Moderate–severe; higher variceal risk', 'CC2D2A-CHF may be underdiagnosed'],
            ['Coloboma frequency', '~15%', '~20–25%', 'Both require fundoscopy at diagnosis'],
            ['Renal involvement', '~38% TIN; ESRD ~20yr', '~38% NPHP11; ESRD ~18–22yr', 'Similar surveillance schedule'],
            ['Retinal dystrophy', '~40% rod-cone', '0% rod-cone (coloboma only)', 'CC2D2A: ERG for rod-cone; TMEM67: fundoscopy only'],
            ['MKS tier', 'MKS6 (biallelic null)', 'MKS3 (biallelic null)', 'Both carry lethal null tier — critical counselling'],
          ]}
        />
        <div className="alert alert-info small mt-2">
          <strong>Clinical implication:</strong> CC2D2A-COACH produces rod-cone dystrophy (~40%) in addition to coloboma (~15%).
          TMEM67-COACH does NOT cause rod-cone dystrophy (TMEM67 not expressed in photoreceptors).
          Distinguish by ERG: CC2D2A → abnormal ERG (rod-cone); TMEM67 → normal ERG (coloboma structural).
          This distinction is critical for retinal monitoring and trial eligibility.
        </div>
      </Section>

      <Section title="CC2D2A Allele-Class Tier Rule — Decision Table" color={ACCENT7}>
        <div className="alert small" style={{ borderLeft: `5px solid ${ACCENT7}`, background: '#fafafa' }}>
          <strong style={{ color: ACCENT7 }}>CC2D2A (JBTS9/MKS6) — Allele-class tier stratification:</strong>
          <ul className="mb-0 mt-1">
            <li>Biallelic null/null → <strong style={{color: ACCENT5}}>MKS6</strong> (perinatal lethal — encephalocele + polydactyly + PKD)</li>
            <li>Null + hypomorphic missense → <strong>JBTS9</strong> (moderate-severe — MTS + variable extra)</li>
            <li>Moderate missense / moderate missense → <strong>JBTS9</strong> (moderate — MTS + hepato-renal)</li>
            <li>Hypomorphic missense / hypomorphic missense → <strong>JBTS9 mild</strong> (MTS + mild ataxia; low extra)</li>
            <li>Rare: biallelic mild missense → <strong>NPHP-like</strong> (pure renal, no MTS — very rare)</li>
          </ul>
          <hr />
          <strong style={{ color: ACCENT6 }}>Genetic counselling implications:</strong>
          <ul className="mb-0 mt-1">
            <li>Parents each carrying one null CC2D2A allele: 25% risk MKS6 lethal, 50% JBTS9 carrier, 25% unaffected</li>
            <li>Prenatal testing (CVS or amniocentesis) recommended; PGT-M available</li>
            <li>Prenatal USS + MRI for subsequent pregnancies with null-carrier parents</li>
          </ul>
        </div>
      </Section>

      <Section title="JSOFD — Post-Axial Polydactyly Subtype (CC2D2A / KIF7 / TCTN)" color={ACCENT9}>
        <DataTable
          headers={['Gene', 'JBTS Type', 'Polydactyly %', 'Orofacial Features', 'Specific Assessment']}
          rows={[
            ['CC2D2A', 'JBTS9', '~20%', 'Macroglossia, hamartomas, accessory frenulae', 'Skeletal survey + orofacial exam'],
            ['KIF7', 'JBTS12', '~35% (highest)', 'Macroglossia, cleft palate risk', 'Skeletal survey + craniofacial team'],
            ['TCTN1', 'JBTS13', '~15%', 'Variable orofacial', 'Skeletal survey'],
            ['TCTN2', 'JBTS24/MKS8', '~25%', 'Variable; MKS8 tier if biallelic null', 'Skeletal + MKS tier counselling'],
            ['ARL13B', 'JBTS8', '~5% (rare)', 'None', 'Skeletal survey if present'],
          ]}
        />
      </Section>
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Section title="Glossary — CC2D2A / JBTS9 Key Terms" color={ACCENT2}>
        <DataTable
          headers={['Term', 'Definition']}
          rows={Object.entries(data).map(([k, v]) => [
            <strong key={k} style={{ color: ACCENT2 }}>{k}</strong>, v,
          ])}
        />
      </Section>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────────
export default function JBTS9Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts9/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts9/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts9/definitions`).then(r => r.json()),
    ]).then(([ov, br, df]) => {
      setOverview(ov);
      setBreakdown(br);
      setDefs(df);
    }).catch(e => setError(e.message));
  }, []);

  return (
    <div>
      {/* Header */}
      <div className="px-4 py-3" style={{ background: ACCENT, color: '#fff' }}>
        <div className="d-flex align-items-center gap-3">
          <Link href="/" className="text-white text-decoration-none">← Home</Link>
          <div>
            <h4 className="mb-0 fw-bold">🧬 CC2D2A — Joubert Syndrome Type 9 (JBTS9) / MKS6</h4>
            <div className="small opacity-75">
              TZ Scaffold / MKS6 Lethal Tier / COACH / Hepato-Renal · 4p15.33 · OMIM Gene *612013 · JBTS9 #612285 · MKS6 #612284 · {N_COHORT}-patient cohort (seed-{SEED})
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="px-4 pt-3">
        <ul className="nav nav-tabs mb-3">
          {TABS.map((t, i) => (
            <li key={i} className="nav-item">
              <button
                className={`nav-link ${tab === i ? 'active fw-bold' : ''}`}
                onClick={() => setTab(i)}
              >{t}</button>
            </li>
          ))}
        </ul>

        {error && (
          <div className="alert alert-danger small">API error: {error}</div>
        )}

        {tab === 0 && <OverviewTab data={overview} />}
        {tab === 1 && <BreakdownTab data={breakdown} />}
        {tab === 2 && <PearlTab data={overview} />}
        {tab === 3 && <DefinitionsTab data={defs} />}
      </div>
    </div>
  );
}
