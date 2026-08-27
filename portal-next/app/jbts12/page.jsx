'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'KIF7 Hedgehog Pearls', 'Definitions'];

// JBTS12 colour scheme — KIF7 / kinesin / GLI / Hedgehog / ACLS / high polydactyly
const ACCENT  = '#1a237e';   // deep indigo — KIF7 / Hedgehog / ciliary tip
const ACCENT2 = '#1565c0';   // strong blue — MTS / neurological
const ACCENT3 = '#4a148c';   // deep purple — GLI processing / Hedgehog
const ACCENT4 = '#1b5e20';   // deep green — curative endpoint / transplant
const ACCENT5 = '#e65100';   // burnt orange — polydactyly (HIGH) / ACLS
const ACCENT6 = '#37474f';   // dark slate — domain matrix
const ACCENT7 = '#b71c1c';   // deep red — retinal / rod-cone
const ACCENT8 = '#bf360c';   // dark orange-red — corpus callosum anomaly
const ACCENT9 = '#00695c';   // dark teal — renal NPHP-like
const ACCENT10= '#880e4f';   // dark pink — no MKS / HLS2 tier distinction

const SEED = 431;
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

// ── Tab: Overview ─────────────────────────────────────────────────────────────
function OverviewTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { kpis = [], hallmark, kif7_function_pearl, no_mks_pearl, polydactyly_cc_pearl,
          first_description, gene_summary, phenotype_summary,
          allele_class_distribution = [] } = data;
  return (
    <div>
      <AlertBox color={ACCENT3} title="⚠ KIF7 — VERTEBRATE COSTAL-2 HOMOLOG: IMMOTILE KINESIN AT CILIARY TIPS CONTROLLING GLI ACTIVATOR/REPRESSOR BALANCE">
        {kif7_function_pearl}
      </AlertBox>
      <AlertBox color={ACCENT5} title="⚠ HIGH POLYDACTYLY (~35-45%) + CORPUS CALLOSUM ANOMALY (~20-25%) — KIF7/JBTS12 ACLS SPECTRUM">
        {polydactyly_cc_pearl}
      </AlertBox>
      <AlertBox color={ACCENT10} title="⚠ NO MKS TIER — KIF7 BIALLELIC NULL → JBTS12 (LIVE BIRTH); ACLS ALLELIC; HLS2 IS DISTINCT LETHAL TIER">
        {no_mks_pearl}
      </AlertBox>

      <Section title="KPI Summary — JBTS12 Cohort (N=40, seed-431)" color={ACCENT2}>
        <div className="row g-2">
          {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
        </div>
      </Section>

      <div className="row">
        <div className="col-md-6">
          <Section title="Gene & Disease Summary" color={ACCENT2}>
            <table className="table table-sm table-bordered small">
              <tbody>
                <tr><td className="fw-bold">Gene</td><td>KIF7 (OMIM *611254)</td></tr>
                <tr><td className="fw-bold">Disease JBTS12/ACLS</td><td>Acrocallosal Syndrome / Joubert Syndrome 12 (OMIM #200990) — Autosomal Recessive</td></tr>
                <tr><td className="fw-bold">HLS2 Tier</td><td>Hydrolethalus Syndrome 2 (OMIM #614120) — severe allelic tier; NOT classic MKS</td></tr>
                <tr><td className="fw-bold">Chromosome</td><td>15q26.1</td></tr>
                <tr><td className="fw-bold">Protein</td><td>1343 aa — Motor domain (kinesin fold; GLI scaffold) / Neck/CC (dimer) / Central CC (regulatory) / C-tail (GLI2/3 interaction; ciliary tip targeting)</td></tr>
                <tr><td className="fw-bold">Inheritance</td><td>Autosomal recessive — biallelic LOF; null/null → severe JBTS12/ACLS; null/missense → moderate; missense/missense → mild/partial ACLS</td></tr>
                <tr><td className="fw-bold">Prevalence</td><td>~1–2% all JBTS; ~1/1,000,000–2,500,000 worldwide</td></tr>
                <tr><td className="fw-bold" style={{ color: ACCENT10 }}>MKS Tier</td><td style={{ color: ACCENT10, fontWeight: 'bold' }}>NONE — KIF7 biallelic null → JBTS12 (live birth); HLS2 is distinct lethal tier (foramen of Monro mechanism, not classic MKS)</td></tr>
                <tr><td className="fw-bold" style={{ color: ACCENT5 }}>Polydactyly</td><td style={{ color: ACCENT5, fontWeight: 'bold' }}>~35-45% — HIGHEST common JBTS rate; bilateral hands+feet; post-axial; ACLS spectrum</td></tr>
                <tr><td className="fw-bold" style={{ color: ACCENT8 }}>Corpus Callosum</td><td style={{ color: ACCENT8, fontWeight: 'bold' }}>~20-25% agenesis/hypoplasia — KIF7 distinctive; ACLS diagnosis when CC anomaly + MTS</td></tr>
                <tr><td className="fw-bold">Retinal</td><td>~18% rod-cone dystrophy; annual ERG (lower than MKS-tier JBTS)</td></tr>
                <tr><td className="fw-bold">Renal</td><td>~12% NPHP-like TIN; ESRD risk; transplant curative (lower than MKS-tier)</td></tr>
                <tr><td className="fw-bold">First Description</td><td>{first_description}</td></tr>
                <tr><td className="fw-bold">Hallmark</td><td>{hallmark}</td></tr>
              </tbody>
            </table>
          </Section>
        </div>
        <div className="col-md-6">
          <Section title="Phenotype Frequencies (N=40)" color={ACCENT6}>
            <DataTable
              headers={['Feature', 'Frequency', 'Notes']}
              rows={[
                ['Molar Tooth Sign (MTS)', '100%', 'Pathognomonic — all JBTS12'],
                ['Cerebellar Ataxia', `${phenotype_summary?.ataxia_pct ?? '–'}%`, 'Core feature; SARA tracking'],
                ['Neonatal Hypotonia', `${phenotype_summary?.hypotonia_pct ?? '–'}%`, 'Feeding difficulty in infancy'],
                ['Oculomotor Apraxia', `${phenotype_summary?.oma_pct ?? '–'}%`, 'Head thrust compensation'],
                ['Intellectual Disability', `${phenotype_summary?.id_pct ?? '–'}%`, 'Corpus callosum anomaly increases risk'],
                ['Breathing Dysregulation', `${phenotype_summary?.breathing_pct ?? '–'}%`, 'Episodic apnea/hyperpnea'],
                ['Polydactyly (post-axial)', `${phenotype_summary?.polydactyly_pct ?? '–'}%`, 'HIGH — bilateral hands+feet; skeletal survey mandatory'],
                ['Corpus Callosum Anomaly', `${phenotype_summary?.corpus_callosum_pct ?? '–'}%`, 'ACLS spectrum; MRI mandatory for CC morphology'],
                ['Retinal Dystrophy (rod-cone)', `${phenotype_summary?.retinal_pct ?? '–'}%', 'Annual ERG; lower than MKS-tier JBTS'],
                ['Renal (NPHP-like TIN)', `${phenotype_summary?.renal_pct ?? '–'}%`, 'Lower risk; annual eGFR/urinalysis'],
                ['Hepatic Fibrosis', `${phenotype_summary?.hepatic_pct ?? '–'}%`, 'Rare; LFTs if suspected'],
                ['No MKS Tier / ACLS', 'JBTS12 live birth', 'HLS2 is distinct lethal tier; ACLS = broader spectrum'],
              ]}
            />
          </Section>
        </div>
      </div>

      <Section title="Allele Class Distribution (Cohort)" color={ACCENT6}>
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

// ── Tab: Diagnostic Breakdown ─────────────────────────────────────────────────
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

      <Section title="Key Pathogenic Variants (KIF7)" color={ACCENT7}>
        <DataTable
          headers={['Variant', 'Domain', 'Effect', 'Population', 'Allele Class', 'Severity', 'Poly Risk', 'CC Risk']}
          rows={key_variants.map(v => [
            <strong key={v.variant} style={{ color: ACCENT7 }}>{v.variant}</strong>,
            v.domain, v.effect, v.population, v.allele_class, v.severity, v.poly_risk, v.cc_risk,
          ])}
        />
      </Section>

      <Section title="Domain → Phenotype Severity Matrix" color={ACCENT6}>
        <DataTable
          headers={['Domain', 'Key Variants', 'Function Lost', 'Severity', 'Poly Risk', 'CC Risk']}
          rows={domain_phenotype_matrix.map(d => [
            d.domain, d.key_variants, d.function_lost, d.severity, d.poly_risk, d.cc_risk,
          ])}
        />
      </Section>

      <Section title="KIF7 → GLI Processing → Hedgehog → MTS + Polydactyly Pathway" color={ACCENT2}>
        <DataTable
          headers={['Step', 'Normal Event', 'Effect When KIF7 Lost']}
          rows={pathway_steps.map(s => [s.step, s.event, s.effect_when_lost])}
        />
      </Section>

      <Section title="Clinical Management" color={ACCENT4}>
        <DataTable
          headers={['Intervention', 'Timing', 'Rationale', 'Level']}
          rows={management.map(m => [m.intervention, m.timing, m.rationale, m.level])}
        />
      </Section>

      <Section title="Per-Patient Table (first 20)" color={ACCENT6}>
        <DataTable
          headers={['ID', 'Sex', 'Ethnicity', 'Allele Class', 'Age Dx', 'MTS', 'Ataxia', 'OMA', 'Poly', 'CC Anomaly', 'Retinal', 'Renal', 'Hepatic', 'ID', 'Breathing']}
          rows={patient_table.map(p => [
            p.id, p.sex, p.ethnicity, p.allele, p.age_dx_yr,
            p.mts, p.ataxia, p.oma, p.poly, p.cc, p.retinal, p.renal, p.hepatic,
            p.id_, p.breathing,
          ])}
        />
      </Section>
    </div>
  );
}

// ── Tab: KIF7 Hedgehog Pearls ─────────────────────────────────────────────────
function PearlTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <AlertBox color={ACCENT3} title="⚠ KIF7 AS VERTEBRATE COSTAL-2: DUAL GLI ACTIVATOR/REPRESSOR CONTROL AT CILIARY TIPS">
        KIF7 is an immotile kinesin that accumulates at ciliary distal tips upon Shh stimulation
        (Costal-2 homolog). It has TWO opposing regulatory functions: (1) POSITIVE — promotes
        GLI2/3 activator (GLIA) processing at ciliary tips with SUFU (essential for Hedgehog
        transcriptional activation); (2) NEGATIVE — suppresses ectopic GLI3 repressor (GLI3R)
        formation at the ciliary base in the basal state (prevents constitutive digit repression).
        Loss of KIF7 → both functions fail: GLIA not formed (Hedgehog transcriptional activation
        lost → cerebellar vermis hypoplasia → MTS) AND ectopic GLI3R accumulates (digit patterning
        repressed → high post-axial polydactyly ~35-45%, one of the highest rates in JBTS series).
        The dual GLI control role is unique to KIF7 among major JBTS genes and explains the
        distinctive combination of MTS + high polydactyly + corpus callosum anomaly in JBTS12.
      </AlertBox>

      <Section title="JBTS12 in Context — JBTS3-12 Series Comparison" color={ACCENT2}>
        <DataTable
          headers={['JBTS Type', 'Gene', 'Inheritance', 'MKS Tier', 'Polydactyly', 'CC Anomaly?', 'Key Feature']}
          rows={[
            ['JBTS12', 'KIF7', 'AR biallelic LOF', 'None (HLS2 = distinct lethal tier)', '⭐ HIGH ~35-45%', 'Yes ~20-25%', 'ACLS allelic; highest poly; Costal-2 GLI dual control'],
            ['JBTS11', 'TCTN1', 'AR biallelic LOF', 'None (JBTS11 only)', '~15%', 'No', 'Tectonic complex lipid gate; TCTN2 MKS8 distinction'],
            ['JBTS10', 'OFD1', 'X-linked', 'None (X-linked)', '~25%', 'No', 'Only X-linked JBTS; RP23 allelic; highest retinal ~55%'],
            ['JBTS9', 'CC2D2A', 'AR biallelic LOF', 'MKS6 (lethal)', '~20%', 'No', 'COACH 2nd most common; hepatic ~25%'],
            ['JBTS8', 'ARL13B', 'AR biallelic LOF', 'None', '~5%', 'No', 'INPP5E trafficking; very low poly; no CHF'],
            ['JBTS7', 'RPGRIP1L', 'AR biallelic LOF', 'MKS5 (lethal)', '~12%', 'No', 'TZ Y-link; European Ala229Thr founder'],
            ['JBTS6', 'TMEM67', 'AR biallelic LOF', 'MKS3 (lethal)', '~8%', 'No', 'COACH most common; FN-III Wnt; hepatic ~30%'],
            ['JBTS5', 'CEP290', 'AR biallelic LOF', 'MKS4 (lethal)', '~5%', 'No', 'Most common JBTS (~10-15%); IVS26 deep intronic'],
            ['JBTS4', 'NPHP1', 'AR biallelic LOF', 'None', '~3%', 'No', 'Pure renal subset; 1q22 deletion'],
            ['JBTS3', 'AHI1', 'AR biallelic LOF', 'None', '~8%', 'No', 'Retinal ~40%; Mid-Eastern Arg830Trp founder'],
          ]}
        />
      </Section>

      <AlertBox color={ACCENT8} title="⚠ CORPUS CALLOSUM ANOMALY IN JBTS12 — ACLS SPECTRUM: MTS + CC AGENESIS + POLYDACTYLY = KIF7">
        Corpus callosum agenesis or hypoplasia in ~20-25% of JBTS12 is a KIF7-distinctive
        feature absent in most other major JBTS types. This reflects KIF7 role in midline
        GLI-dependent forebrain patterning — corpus callosum axon guidance requires proper
        Hedgehog signalling. When a patient presents with MTS + polydactyly + corpus callosum
        anomaly, the clinical triad is highly specific for JBTS12/ACLS (KIF7). Brain MRI must
        specifically assess corpus callosum morphology in all JBTS12 patients. The presence
        of corpus callosum anomaly triggers ACLS diagnosis label (OMIM #200990) — the same
        molecular diagnosis as JBTS12, but a broader syndromic description reflecting the
        full KIF7 phenotypic spectrum with forebrain involvement.
      </AlertBox>

      <Section title="KIF7 vs Other JBTS Genes — GLI/Hedgehog Module Interactions" color={ACCENT3}>
        <DataTable
          headers={['Gene', 'JBTS Type', 'Role in Hedgehog/GLI', 'Polydactyly Rate', 'Mechanism']}
          rows={[
            ['KIF7', 'JBTS12', 'Ciliary tip GLI activator + repressor control (Costal-2 homolog)', '⭐ ~35-45%', 'GLI3R excess (ectopic) → digit repression; GLIA loss → MTS; CC anomaly from midline GLI failure'],
            ['INPP5E', 'JBTS1', 'PI(4,5)P2 hydrolysis in cilia → SMO/GLI access', '~10%', 'Lipid gate axis; upstream of KIF7 GLI processing'],
            ['ARL13B', 'JBTS8', 'INPP5E trafficking to cilia; PI4P ciliary identity', '~5%', 'Lipid axis; KIF7 downstream; polydactyly low (INPP5E partial)'],
            ['CEP290', 'JBTS5', 'TZ Y-link inner plate; SMO/GLI import gate', '~5%', 'Gate mechanism; not GLI-specific; polydactyly low'],
            ['TMEM67', 'JBTS6', 'Tectonic domain; MKS module; FN-III Wnt crosstalk', '~8%', 'Gate mechanism; Wnt also dysregulated'],
            ['GLI3', 'Non-JBTS', 'GLI3 transcription factor itself; Greig syndrome/Pallister-Hall (GOF/LOF)', 'Very high (Greig)', 'Downstream of KIF7; GLI3 mutations = digit phenotype without MTS'],
          ]}
        />
        <div className="alert alert-info small mt-2">
          <strong>Key insight:</strong> KIF7 controls GLI3 repressor formation upstream of GLI3 itself. GLI3 LOF
          (Greig syndrome) causes polydactyly without MTS; KIF7 LOF causes polydactyly + MTS because KIF7 controls
          BOTH GLI3R repression AND GLI activator processing — the dual control that makes JBTS12 unique in the series.
        </div>
      </Section>

      <AlertBox color={ACCENT10} title="⚠ NO MKS TIER — KIF7 vs MKS-TIER JBTS GENES: HLS2 IS NOT CLASSIC MKS">
        KIF7 biallelic null → JBTS12 (live birth, Joubert features + ACLS spectrum).
        Hydrolethalus Syndrome 2 (HLS2, OMIM #614120) is the severe allelic tier for KIF7, but
        HLS2 is mechanistically distinct from classic Meckel-Gruber Syndrome (MKS). HLS2 is caused
        by specific KIF7 alleles producing severe forebrain/foramen-of-Monro defects (hydrocephalus
        + foramen obstruction + lethal CNS malformations) — NOT the classic MKS triad of posterior
        encephalocele + polycystic kidneys + extra digits. For reproductive counselling:
        KIF7 null carriers face JBTS12/ACLS or HLS2 risk (distinct from MKS);
        HLS2 is lethal but through a different CNS mechanism than encephalocele-based MKS.
        This distinction affects counselling: KIF7 null families should be counselled about
        HLS2 specifically, not MKS, and the clinical spectrum differs.
      </AlertBox>

      <Section title="Polydactyly Phenotype in JBTS12 — Clinical Characterisation" color={ACCENT5}>
        <DataTable
          headers={['Feature', 'KIF7/JBTS12', 'Other JBTS Types', 'Clinical Significance']}
          rows={[
            ['Penetrance', '~35-45%', 'JBTS9 ~20%, JBTS11 ~15%, JBTS8 ~5%', 'Highest among common JBTS — GLI3R excess mechanism'],
            ['Distribution', 'Bilateral hands AND feet (post-axial)', 'Usually unilateral or hands-only', 'Bilateral B+F involvement characteristic of KIF7/ACLS'],
            ['Type', 'Post-axial (5th digit ray); occasionally preaxial (ACLS)', 'Post-axial only', 'Preaxial component suggests ACLS spectrum'],
            ['Surgery', 'Surgical correction usually feasible', 'Same', 'Orthopaedic surgical planning from infancy; functional outcome good'],
            ['Skeletal survey', 'Mandatory when polydactyly present', 'Mandatory when present', 'Rule out additional skeletal anomalies; characterise polydactyly type'],
            ['Molecular implication', 'Polydactyly + MTS → KIF7 first differential', 'Other polydactyly-JBTS: CC2D2A, TCTN1', 'Add corpus callosum MRI assessment when KIF7 suspected'],
          ]}
        />
      </Section>

      <Section title="Navigation — Adjacent Joubert Syndrome Dashboards" color={ACCENT2}>
        <ul className="list-unstyled small">
          <li><Link href="/jbts11" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS11 (TCTN1) — Tectonic Complex / Lipid Gate / No MKS Tier</Link></li>
          <li><Link href="/jbts10" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS10 (OFD1) — X-linked / Centriolar Satellite / RP23 Allelic</Link></li>
          <li><Link href="/jbts9" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS9 (CC2D2A) — TZ Scaffold / MKS6 / COACH</Link></li>
          <li><Link href="/jbts8" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS8 (ARL13B) — INPP5E Trafficking / Lipid Axis</Link></li>
          <li><Link href="/jbts7" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS7 (RPGRIP1L) — TZ Y-Link / MKS5</Link></li>
          <li><Link href="/jbts6" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS6 (TMEM67) — MKS3 / COACH / Hepatic</Link></li>
          <li><Link href="/jbts5" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS5 (CEP290) — MKS4 / Most Common JBTS</Link></li>
          <li><Link href="/jbts4" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS4 (NPHP1) — Renal Dominant</Link></li>
          <li><Link href="/jbts3" className="text-decoration-none" style={{ color: ACCENT2 }}>← JBTS3 (AHI1) — Retinal Dominant</Link></li>
        </ul>
      </Section>
    </div>
  );
}

// ── Tab: Definitions ──────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Section title="Glossary — KIF7 / JBTS12 / ACLS Key Terms" color={ACCENT2}>
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

// ── Main page ─────────────────────────────────────────────────────────────────
export default function JBTS12Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview]   = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [defs, setDefs]           = useState(null);
  const [error, setError]         = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts12/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts12/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts12/definitions`).then(r => r.json()),
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
            <h4 className="mb-0 fw-bold">🧬 KIF7 — Joubert Syndrome Type 12 (JBTS12) / Acrocallosal Syndrome / Costal-2 Homolog / GLI Dual Control</h4>
            <div className="small opacity-75">
              AR Kinesin GLI Regulator / ACLS Allelic / No MKS Tier · 15q26.1 · OMIM Gene *611254 · JBTS12/ACLS #200990 · HLS2 #614120 · {N_COHORT}-patient cohort (seed-{SEED})
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
