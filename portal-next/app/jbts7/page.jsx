'use client';
import { useState, useEffect } from 'react';
import Link from 'next/link';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const TABS = ['Overview', 'Diagnostic Breakdown', 'Allele-Tier & Y-Link Pearl', 'Definitions'];

// JBTS7 colour scheme — RPGRIP1L / TZ Y-link scaffold / European Ala229Thr / MKS5 lethal
const ACCENT  = '#0d47a1';   // deep blue — JBTS/MTS neurological
const ACCENT2 = '#880e4f';   // deep crimson — MKS5 lethal / biallelic null
const ACCENT3 = '#00695c';   // teal — renal NPHP8
const ACCENT4 = '#1b5e20';   // deep green — transplant outcomes (curative)
const ACCENT5 = '#e65100';   // burnt orange — European Ala229Thr founder pearl
const ACCENT6 = '#4a148c';   // deep purple — allele-class tier rule
const ACCENT7 = '#f57f17';   // amber — RH domain / Y-link / NPHP4 binding
const ACCENT8 = '#37474f';   // dark slate — cerebellar ataxia / OMA
const ACCENT9 = '#b71c1c';   // deep red — retinal / C2-RPGR axis

const SEED = 421;
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
  const { kpis = [], hallmark, critical_diagnostic_pearl,
          allele_phenotype_rule, prevalence, first_description, gene_summary, phenotype_summary } = data;
  return (
    <div>
      <AlertBox color={ACCENT5} title="⚠ CRITICAL DIAGNOSTIC PEARL — EUROPEAN Ala229Thr FOUNDER ALLELE (c.685G>A)">
        {critical_diagnostic_pearl}
      </AlertBox>
      <AlertBox color={ACCENT6} title="🧬 ALLELE-CLASS → DISEASE TIER RULE">
        {allele_phenotype_rule}
      </AlertBox>
      <AlertBox color={ACCENT2} title="☠ MKS5 LETHAL SPECTRUM — Biallelic NULL RPGRIP1L">
        Biallelic truncating/splice-null RPGRIP1L → Meckel-Gruber Syndrome Type 5 (MKS5):
        encephalocele + polydactyly + polycystic kidneys + oligohydramnios. Perinatal lethal.
        NEVER assign MKS5-tier to biallelic p.Ala229Thr (hypomorphic) patients.
      </AlertBox>

      <div className="row mb-3">
        {kpis.map((k, i) => <KPI key={i} label={k.label} value={k.value} color={k.color} />)}
      </div>

      <Section title="Molecular Hallmark" color={ACCENT}>
        <p className="small">{hallmark}</p>
      </Section>

      {phenotype_summary && (
        <Section title="Phenotype Frequencies (n=40 cohort)" color={ACCENT8}>
          <DataTable
            headers={['Feature', 'n', '%', 'Note']}
            rows={[
              ['Molar Tooth Sign', N_COHORT * phenotype_summary.mts_pct / 100 | 0, `${phenotype_summary.mts_pct}%`, 'Pathognomonic; brain MRI mandatory'],
              ['Cerebellar Ataxia', N_COHORT * phenotype_summary.ataxia_pct / 100 | 0, `${phenotype_summary.ataxia_pct}%`, 'Gait ataxia; truncal instability'],
              ['Neonatal Hypotonia', N_COHORT * phenotype_summary.hypotonia_pct / 100 | 0, `${phenotype_summary.hypotonia_pct}%`, 'Universal early feature'],
              ['Oculomotor Apraxia', N_COHORT * phenotype_summary.oma_pct / 100 | 0, `${phenotype_summary.oma_pct}%`, 'Horizontal gaze failure; ~45%'],
              ['Intellectual Disability', N_COHORT * phenotype_summary.id_pct / 100 | 0, `${phenotype_summary.id_pct}%`, 'Moderate > severe'],
              ['Breathing Dysregulation', N_COHORT * phenotype_summary.breathing_pct / 100 | 0, `${phenotype_summary.breathing_pct}%`, 'Neonatal episodic apnea; self-resolves'],
              ['Retinal Dystrophy', N_COHORT * phenotype_summary.retinal_pct / 100 | 0, `${phenotype_summary.retinal_pct}%`, 'Rod-cone; C2-RPGR axis; annual ERG'],
              ['Renal NPHP8', N_COHORT * phenotype_summary.renal_pct / 100 | 0, `${phenotype_summary.renal_pct}%`, 'TIN; ESRD median ~22yr'],
              ['Hepatic (mild CHF)', N_COHORT * phenotype_summary.hepatic_pct / 100 | 0, `${phenotype_summary.hepatic_pct}%`, 'Mild CHF only — NOT COACH gene'],
              ['Polydactyly', N_COHORT * phenotype_summary.polydactyly_pct / 100 | 0, `${phenotype_summary.polydactyly_pct}%', 'Postaxial; rare in JBTS7'],
              ['European Ala229Thr', N_COHORT * phenotype_summary.ala229_pct / 100 | 0, `${phenotype_summary.ala229_pct}%`, 'Most common allele; European enriched'],
            ]}
          />
        </Section>
      )}

      {gene_summary && (
        <Section title="Gene & Disease Summary" color={ACCENT7}>
          <DataTable
            headers={['Field', 'Value']}
            rows={[
              ['Symbol', gene_summary.symbol],
              ['Alias', gene_summary.alias],
              ['OMIM Gene', gene_summary.omim_gene],
              ['OMIM JBTS7', gene_summary.omim_disease_jbts7],
              ['OMIM MKS5', gene_summary.omim_disease_mks5],
              ['OMIM NPHP8', gene_summary.omim_disease_nphp8],
              ['Chromosome', gene_summary.chromosome],
              ['Protein', gene_summary.protein_length],
              ['Protein Class', gene_summary.protein_class],
              ['Function', gene_summary.function],
            ]}
          />
        </Section>
      )}

      <Section title="First Description" color={ACCENT8}>
        <p className="small text-muted">{first_description}</p>
      </Section>
      <Section title="Prevalence" color={ACCENT8}>
        <p className="small text-muted">{prevalence}</p>
      </Section>
    </div>
  );
}

// ── Tab: Diagnostic Breakdown ─────────────────────────────────────────────────
function BreakdownTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  const { ethnicity = [], allele_class = [], age_at_diagnosis = [],
          phenotype_matrix = [], transplant_outcomes, allele_protein_table = [],
          jbts_comparison = [] } = data;
  return (
    <div>
      <Section title="Ethnicity Distribution" color={ACCENT}>
        <DataTable
          headers={['Ethnicity', 'n', '%']}
          rows={ethnicity.map(e => [e.ethnicity, e.n, `${e.pct}%`])}
        />
      </Section>

      <Section title="Allele Class Distribution" color={ACCENT6}>
        <DataTable
          headers={['Allele Class', 'n', '%']}
          rows={allele_class.map(a => [a.class, a.n, `${a.pct}%`])}
        />
      </Section>

      <Section title="Age at Diagnosis" color={ACCENT8}>
        <DataTable
          headers={['Age Bucket', 'n', '%']}
          rows={age_at_diagnosis.map(a => [a.bucket, a.n, `${a.pct}%`])}
        />
      </Section>

      <Section title="Phenotype Matrix" color={ACCENT7}>
        <DataTable
          headers={['Feature', 'n', '%', 'Note']}
          rows={phenotype_matrix.map(p => [p.feature, p.n, `${p.pct}%`, p.note])}
        />
      </Section>

      {transplant_outcomes && (
        <Section title="Renal Transplant Outcomes (NPHP8)" color={ACCENT3}>
          <DataTable
            headers={['Metric', 'Value']}
            rows={[
              ['Estimated renal Tx needed', transplant_outcomes.n_renal_transplant_needed],
              ['Renal Tx outcome', transplant_outcomes.renal_tx_outcome],
              ['Neurological corrected?', transplant_outcomes.neurological_corrected],
              ['Retinal corrected?', transplant_outcomes.retinal_corrected],
              ['Mean ESRD age (est.)', transplant_outcomes.avg_esrd_age ? `${transplant_outcomes.avg_esrd_age}yr` : 'N/A'],
              ['Liver Tx indicated?', 'No — not a COACH gene'],
            ]}
          />
          <div className="small text-muted mt-1">{transplant_outcomes.note}</div>
        </Section>
      )}

      <Section title="Allele → Protein → Phenotype Table" color={ACCENT2}>
        <DataTable
          headers={['Variant', 'cDNA', 'Domain', 'Ethnicity', 'Tier']}
          rows={allele_protein_table.map(a => [a.variant, a.cdna, a.domain, a.ethnic, a.tier])}
        />
      </Section>

      {jbts_comparison.length > 0 && (
        <Section title="JBTS Type Comparison" color={ACCENT8}>
          <DataTable
            headers={['Type', 'Gene', 'Distinctive Feature', 'COACH?', 'Hepatic?']}
            rows={jbts_comparison.map(j => [j.type, j.gene, j.distinctive, j.coach, j.hepatic])}
          />
        </Section>
      )}
    </div>
  );
}

// ── Tab: Allele-Tier & Y-Link Pearl ──────────────────────────────────────────
function AlleleTierTab({ overview, breakdown }) {
  return (
    <div>
      <AlertBox color={ACCENT5} title="⚡ EUROPEAN Ala229Thr FOUNDER — Most Common RPGRIP1L Allele">
        p.Ala229Thr (c.685G>A) is the most common RPGRIP1L allele in European populations.
        It disrupts the N-terminal coiled-coil domain partially — a HYPOMORPHIC allele.
        Biallelic p.Ala229Thr → mild JBTS7 (MTS + mild neurological). NOT MKS5-tier.
        Present at low gnomAD frequency; MTS on brain MRI confirms pathogenicity.
        Compound het Ala229Thr + NULL → JBTS7 moderate (most common European genotype).
      </AlertBox>

      <AlertBox color={ACCENT7} title="🔬 TZ Y-LINK SCAFFOLD — RPGRIP1L Molecular Function">
        RPGRIP1L anchors the Y-link spokes of the ciliary transition zone (TZ) to the axonemal
        doublet microtubules via its RH domain (NPHP4 binding) and C2 domain (membrane + RPGR binding).
        Loss → Y-link detachment → GPCR/Smoothened exclusion from cilium → Hedgehog failure
        → Molar Tooth Sign. The C2-RPGR interaction explains variable retinal involvement in JBTS7
        (retinal dystrophy ~25%) without the high retinal rate of JBTS5 (CEP290, ~57%).
      </AlertBox>

      <AlertBox color={ACCENT2} title="☠ MKS5 BIALLELIC NULL LETHAL SPECTRUM">
        Biallelic truncating (Lys1326Ter/Lys1326Ter; Trp519Ter/any-null) → MKS5: Meckel-Gruber
        Syndrome Type 5. Perinatal lethal: encephalocele, polydactyly, polycystic kidneys,
        oligohydramnios. This is a PRENATAL DIAGNOSIS scenario — ultrasound + molecular.
        RPGRIP1L biallelic null is NOT compatible with extrauterine life.
      </AlertBox>

      <Section title="Allele-Class → Disease Tier Rule" color={ACCENT6}>
        <DataTable
          headers={['Genotype Class', 'Disease Tier', 'MTS?', 'Renal?', 'MKS5 Lethal?']}
          rows={[
            ['Biallelic NULL (2 truncating/splice)', 'MKS5 — lethal', 'N/A (perinatal lethal)', 'PKD (lethal)', 'YES'],
            ['One NULL + one HYPOMORPHIC (e.g. Ala229Thr)', 'JBTS7 moderate', 'Yes (~87%)', 'NPHP8 ~30%', 'No'],
            ['Biallelic Ala229Thr (European)', 'JBTS7 mild', 'Yes (~87%)', 'NPHP8 ~25%', 'No'],
            ['Biallelic HYPOMORPHIC (missense/missense)', 'JBTS7 mild / NPHP8', 'Yes (~80%)', 'NPHP8 ~28%', 'No'],
            ['Asn694Ser compound het (South Asian)', 'NPHP8 renal-dominant', 'Mild or absent', 'Yes (dominant)', 'No'],
          ]}
        />
      </Section>

      <Section title="Domain-Variant Correlation" color={ACCENT}>
        <DataTable
          headers={['Domain', 'Key Variant', 'Mechanism', 'Phenotype Impact']}
          rows={[
            ['N-terminal CC (aa 1-450)', 'Ala229Thr', 'Partial CC fold destabilisation; hypomorphic', 'JBTS7 mild (European founder)'],
            ['CC/RH boundary', 'Trp519Ter', 'Truncating null; biallelic → MKS5', 'MKS5 lethal if biallelic null'],
            ['RH domain (aa 450-960)', 'Arg1174Gln', 'NPHP4 interface disrupted; Y-link partial fail', 'JBTS7 / NPHP8 (MENA)'],
            ['RH domain core', 'Asn694Ser', 'RH fold partial; NPHP4 binding reduced', 'NPHP8 renal-dominant (South Asian)'],
            ['RH domain (splice)', 'c.2407+2T>A', 'Null (splice donor exon 15); European', 'Null; JBTS7 compound het'],
            ['RH-C2 junction', 'Leu821Pro', 'Disrupts RH-C2 hinge; MENA', 'JBTS7 moderate'],
            ['C2 domain (aa 960-1315)', 'Lys1326Ter', 'C2 truncating null; RPGR binding lost', 'Null; biallelic → MKS5'],
          ]}
        />
      </Section>

      <Section title="JBTS7 vs JBTS6 — No COACH in JBTS7" color={ACCENT3}>
        <p className="small">
          JBTS7 (RPGRIP1L) is <strong>NOT a COACH gene</strong>. Hepatic fibrosis in JBTS7 is
          mild CHF only (~10%), without the ductal plate malformation and portal hypertension
          seen in JBTS6 (TMEM67) and JBTS9 (CC2D2A). Annual liver ultrasound is prudent
          but combined liver-kidney transplant is NOT indicated in JBTS7. Only TMEM67 and
          CC2D2A carry the COACH syndrome designation.
        </p>
        <p className="small">
          Retinal dystrophy in JBTS7 (~25%) is lower than JBTS5 (CEP290, ~57%) — the C2-domain
          RPGR interaction explains partial retinal involvement. Annual ERG is mandatory but
          rod-cone progression is typically slower than CEP290-associated retinopathy.
        </p>
      </Section>
    </div>
  );
}

// ── Tab: Definitions ─────────────────────────────────────────────────────────
function DefinitionsTab({ data }) {
  if (!data) return <div className="text-muted">Loading…</div>;
  return (
    <div>
      <Section title="Gene & Protein" color={ACCENT}>
        <DataTable
          headers={['Term', 'Definition']}
          rows={[
            ['Gene', data.gene],
            ['OMIM Gene', data.omim_gene],
            ['OMIM JBTS7', data.omim_jbts7],
            ['OMIM MKS5', data.omim_mks5],
            ['OMIM NPHP8', data.omim_nphp8],
            ['Chromosome', data.chromosome],
            ['Protein', data.protein],
            ['Pathway', data.pathway],
          ]}
        />
      </Section>
      <Section title="Disease Definitions" color={ACCENT6}>
        <DataTable
          headers={['Term', 'Definition']}
          rows={[
            ['Allele Tier Rule', data.allele_tier_rule],
            ['MKS5', data.mks5],
            ['NPHP8', data.nphp8],
            ['MTS', data.mts],
            ['OMA', data.oma],
          ]}
        />
      </Section>
      <Section title="Molecular Concepts" color={ACCENT7}>
        <DataTable
          headers={['Term', 'Definition']}
          rows={[
            ['Ala229Thr Pearl', data.ala229thr_pearl],
            ['RH Domain', data.rh_domain],
            ['C2 Domain', data.c2_domain],
            ['Y-Link Scaffold', data.y_link_scaffold],
            ['TZ Module', data.tz_module],
            ['No COACH', data.no_coach],
          ]}
        />
      </Section>
      <Section title="Management & Context" color={ACCENT4}>
        <DataTable
          headers={['Term', 'Definition']}
          rows={[
            ['Therapy Status', data.therapy_status],
            ['Inheritance', data.inheritance],
            ['Frequency', data.frequency],
            ['Related Genes', data.related_genes],
          ]}
        />
      </Section>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function JBTS7Page() {
  const [tab, setTab] = useState(0);
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/jbts7/overview`).then(r => r.json()),
      fetch(`${API}/api/jbts7/breakdown`).then(r => r.json()),
      fetch(`${API}/api/jbts7/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); })
      .catch(e => setError(e.message));
  }, []);

  return (
    <div style={{ display: 'flex', minHeight: '100vh', fontFamily: 'system-ui, sans-serif' }}>
      {/* ── left nav ── */}
      <nav style={{
        width: 210, minWidth: 210, background: '#0d47a1',
        color: '#fff', padding: '1.25rem 0.9rem', display: 'flex',
        flexDirection: 'column', overflowY: 'auto',
      }}>
        <div className="fw-bold mb-1" style={{ fontSize: '0.85rem' }}>JBTS7 — RPGRIP1L</div>
        <div style={{ fontSize: '0.7rem', color: '#90caf9' }} className="mb-3">
          FTM / NPHP8 / MKS5<br />16q12.2 · 1315 aa · OMIM #611560
        </div>
        <ul className="list-unstyled mb-3" style={{ fontSize: '0.78rem' }}>
          {TABS.map((t, i) => (
            <li key={i} className="mb-1">
              <button
                onClick={() => setTab(i)}
                className="btn btn-sm w-100 text-start"
                style={{ background: tab === i ? '#ffffff22' : 'transparent', color: '#fff', border: 'none' }}
              >
                {t}
              </button>
            </li>
          ))}
        </ul>
        <hr style={{ borderColor: '#ffffff44' }} />
        <div className="small mb-2 fw-bold" style={{ color: '#90caf9' }}>JBTS Series</div>
        <ul className="list-unstyled" style={{ fontSize: '0.72rem' }}>
          <li><Link href="/jbts3" className="text-white-50">← JBTS3 (AHI1)</Link></li>
          <li><Link href="/jbts4" className="text-white-50">← JBTS4 (NPHP1)</Link></li>
          <li><Link href="/jbts5" className="text-white-50">← JBTS5 (CEP290)</Link></li>
          <li><Link href="/jbts6" className="text-white-50">← JBTS6 (TMEM67)</Link></li>
          <li className="fw-bold" style={{ color: '#fff' }}>▶ JBTS7 (RPGRIP1L)</li>
        </ul>
        <hr style={{ borderColor: '#ffffff44' }} />
        <div className="small mb-2 fw-bold" style={{ color: '#90caf9' }}>Related</div>
        <ul className="list-unstyled" style={{ fontSize: '0.72rem' }}>
          <li><Link href="/nphp8" className="text-white-50">NPHP8 (RPGRIP1L renal)</Link></li>
          <li><Link href="/joubert" className="text-white-50">Joubert Syndrome overview</Link></li>
          <li><Link href="/jbts9" className="text-white-50">JBTS9 (CC2D2A) →</Link></li>
        </ul>
        <hr style={{ borderColor: '#ffffff44' }} />
        <div className="small text-white-50">
          Cohort: {N_COHORT} patients · seed {SEED}
          <br />OMIM: #611560 · Gene: *610937
          <br />No COACH · Y-link scaffold
        </div>
      </nav>

      {/* ── main content ── */}
      <main style={{ flex: 1, padding: '1.5rem', overflowY: 'auto' }}>
        <div className="d-flex align-items-center mb-1">
          <h4 className="mb-0 me-3" style={{ color: ACCENT }}>
            RPGRIP1L — Joubert Syndrome Type 7 (JBTS7)
          </h4>
          <span className="badge" style={{ background: ACCENT5, fontSize: '0.75rem' }}>
            European Ala229Thr Founder · Y-Link Scaffold · MKS5 Biallelic Null
          </span>
        </div>
        <div className="text-muted small mb-3">
          FTM · NPHP8 · MKS5 · 16q12.2 · 1315 aa · OMIM #611560 · seed {SEED} · {N_COHORT} patients
        </div>

        {error && (
          <div className="alert alert-danger">API error: {error}</div>
        )}

        {/* Tab bar */}
        <ul className="nav nav-tabs mb-4">
          {TABS.map((t, i) => (
            <li key={i} className="nav-item">
              <button
                className={`nav-link ${tab === i ? 'active' : ''}`}
                onClick={() => setTab(i)}
              >
                {t}
              </button>
            </li>
          ))}
        </ul>

        {tab === 0 && <OverviewTab data={overview} />}
        {tab === 1 && <BreakdownTab data={breakdown} />}
        {tab === 2 && <AlleleTierTab overview={overview} breakdown={breakdown} />}
        {tab === 3 && <DefinitionsTab data={definitions} />}
      </main>
    </div>
  );
}
