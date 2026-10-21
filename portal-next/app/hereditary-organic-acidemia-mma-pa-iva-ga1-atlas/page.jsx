'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';
const SLUG = 'hereditary-organic-acidemia-mma-pa-iva-ga1-atlas';
const TABS = ['Overview', 'Gene Table', 'OA Groups & NBS Pathways', 'Definitions'];

const GENE_COLORS = {
  'MUT':    '#1a237e',  // deep navy        — AR; isolated MMA; C3; mut0 no B12; metabolic stroke
  'MMACHC': '#b71c1c',  // deep red         — AR; cblC; combined MMA+HCU; HUS neonatal; retinal degen
  'PCCA':   '#4a148c',  // deep violet      — AR; PA alpha; methylcitric; cardiomyopathy
  'PCCB':   '#880e4f',  // dark magenta     — AR; PA beta; identical PCCA; Inuit founder
  'IVD':    '#2e7d32',  // dark green       — AR; IVA; sweaty feet C5; A282V mild NBS
  'GCDH':   '#e65100',  // deep orange      — AR; GA1; macrocephaly; bat-wing MRI; fever protocol
  'HMGCL':  '#00695c',  // dark teal        — AR; HMG-CoA lyase; hypoketotic hypoglycemia; fasting CI
  'ACAT1':  '#1565c0',  // deep blue        — AR; T2/3-oxothiolase; episodic ketoacidosis
};

const GENE_INFO = {
  'MUT':    { full: 'MUT / Methylmalonyl-CoA Mutase / Isolated MMA / 750aa', locus: '6p12.3', inh: 'AR LOF',
               note: '750aa/83kDa; AdoCbl-dependent MCM; propionyl-CoA catabolism → succinyl-CoA; LOF → MMA accumulates; C3 on NBS; mut0 (no enzyme) = severe neonatal, NO B12 response, liver transplant; mut- (residual) = partial B12 response; METABOLIC STROKE: globus pallidus infarction; renal tubular acidosis → progressive CKD; 3-OH-propionic acid but NO methylcitric acid (DDx PA)' },
  'MMACHC': { full: 'MMACHC / cblC Chaperone / Combined MMA+Homocystinuria / 282aa', locus: '1p34.1', inh: 'AR LOF',
               note: '282aa/32kDa; intracellular cobalamin processing step 1; LOF → neither AdoCbl NOR MeCbl can be made → combined MMA+Hcy; c.271dupA most common European allele; NEONATAL HUS (TMA) = PATHOGNOMONIC early-onset; RETINAL DEGENERATION (cone-rod dystrophy) progressive; C3 on NBS + low methionine; OHCbl IM daily + betaine + folate; NOT cyanocobalamin; annual retinal surveillance MANDATORY' },
  'PCCA':   { full: 'PCCA / Propionyl-CoA Carboxylase α / Propionic Acidemia Type A / 728aa', locus: '13q32.3', inh: 'AR LOF',
               note: '728aa/80kDa; α-subunit of biotin-dependent PCC (α6β6 dodecamer); propionyl-CoA → methylmalonyl-CoA; LOF → 3-OH-PROPIONIC ACID + METHYLCITRIC ACID = PATHOGNOMONIC for PA; C3 on NBS; secondary hyperammonemia (inhibits NAGS → urea cycle); DILATED CARDIOMYOPATHY up to 30% = LEADING CAUSE DEATH; ECHO MANDATORY annually; liver transplant prevents crises' },
  'PCCB':   { full: 'PCCB / Propionyl-CoA Carboxylase β / Propionic Acidemia Type B / 539aa', locus: '3q22.3', inh: 'AR LOF',
               note: '539aa/58kDa; β-subunit of PCC; PHENOTYPICALLY IDENTICAL to PCCA — only genotype distinguishes; IVS12+3A>G (c.1218+3A>G) = Inuit/Greenlandic/Arctic founder allele; same metabolic markers (3-OH-propionic + methylcitric); same cardiomyopathy risk; same annual echo mandatory; gene panel required to distinguish PCCA vs PCCB' },
  'IVD':    { full: 'IVD / Isovaleryl-CoA Dehydrogenase / Isovaleric Acidemia / 394aa', locus: '15q15.1', inh: 'AR LOF',
               note: '394aa/46kDa; FAD-dependent mitochondrial enzyme; leucine catabolism step; LOF → isovaleric acid accumulates; SWEATY FEET / CHEESY ODOR = PATHOGNOMONIC (volatile isovaleric acid); C5 (ISOVALERYLCARNITINE) ELEVATED ON NBS = PATHOGNOMONIC for IVD; A282V MILD VARIANT: most NBS-identified, nearly asymptomatic — clinical significance UNCERTAIN; glycine + L-carnitine conjugation therapy; exclude pivalate drugs (pivampicillin)' },
  'GCDH':   { full: 'GCDH / Glutaryl-CoA Dehydrogenase / Glutaric Aciduria Type 1 / 438aa', locus: '19p13.13', inh: 'AR LOF',
               note: '438aa/48kDa; Lys/Hyl/Trp catabolism; LOF → GA + 3-OH-GA accumulate → NEUROTOXIC to striatum; MACROCEPHALY (>97th centile) at birth = HIGHLY CHARACTERISTIC; BAT-WING FRONTOTEMPORAL ATROPHY on MRI = PATHOGNOMONIC; WINDOW OF VULNERABILITY 3-36 months: fever triggers STRIATAL NECROSIS → permanent acute dystonia; LOW EXCRETOR TRAP: C5DC may be borderline; EMERGENCY FEVER PROTOCOL = glucose IV + carnitine during ANY fever; p.Arg402Trp 50% European alleles' },
  'HMGCL':  { full: 'HMGCL / HMG-CoA Lyase / HMG-CoA Lyase Deficiency / 325aa', locus: '1p36.11', inh: 'AR LOF',
               note: '325aa/34kDa; final leucine catabolism step + rate-limiting ketogenesis step; LOF → ketones CANNOT be made + 3-HMG accumulates; HYPOKETOTIC HYPOGLYCEMIA = PATHOGNOMONIC (glucose LOW + ketones ABSENT = brain has NO backup); 3-HMG + 3-methylglutaconic in urine; C6DC on NBS; FASTING ABSOLUTELY CONTRAINDICATED; IV glucose 10% IMMEDIATELY in any crisis; Saudi Arabia p.Arg41Gln founder; DDx MCAD: MCAD lacks 3-HMG' },
  'ACAT1':  { full: 'ACAT1 / Mitochondrial Acetoacetyl-CoA Thiolase / 3-Oxothiolase (T2) / 427aa', locus: '11q22.3', inh: 'AR LOF',
               note: '427aa/45kDa; isoleucine catabolism + ketone body utilization; LOF → 2-methyl-3-OH-butyric acid + tiglylglycine accumulate during catabolic stress; EPISODIC SEVERE KETOACIDOSIS with NORMAL INTERVALS = PATHOGNOMONIC; pH can be <6.8 but aggressive NaHCO3 + glucose → FULL RECOVERY; C5-OH + C5:1 on NBS; 2M3HBA in urine; excellent long-term prognosis; avoid prolonged fasting; DDx HMGCL: ACAT1 = HYPERketotic (opposite)' },
};

const PATHOGNOMONIC = {
  'MUT':    'ISOLATED MMA — C3 (propionylcarnitine) elevated + urine MMA markedly elevated. NO methylcitric acid (key DDx from PA). mut0 vs mut-: B12 loading test — mut0 = NO response; mut- = partial response. METABOLIC STROKE: acute dystonia + MRI basal ganglia infarction. Renal tubular acidosis → CKD. Liver transplant = metabolic cure for mut0 (renal disease may still progress).',
  'MMACHC': 'cblC (COMBINED MMA + HCU) — PATHOGNOMONIC: neonatal HUS (hemolysis + thrombocytopenia + renal failure = TMA) in early-onset cblC. C3 elevated + total plasma Hcy ELEVATED + methionine LOW on NBS. ALL C3-positive NBS MUST have homocysteine measured to exclude cblC. RETINAL DEGENERATION is progressive even in well-treated patients — annual retinal exam MANDATORY. OHCbl (NOT cyanocobalamin) + betaine + folate.',
  'PCCA':   'PROPIONIC ACIDEMIA — PATHOGNOMONIC: 3-OH-propionic acid + METHYLCITRIC ACID in urine (methylcitric acid ABSENT in MMA). C3 on NBS; secondary hyperammonemia (NH3 may be >1000 µmol/L). DILATED CARDIOMYOPATHY: develops in up to 30% even on diet — ECHOCARDIOGRAM EVERY 1-2 YEARS mandatory. DCM is leading cause of death in PA. LIVER TRANSPLANT corrects metabolic risk + may prevent further DCM progression.',
  'PCCB':   'PA TYPE B — IDENTICAL to PCCA clinically and biochemically. PCCB-specific: IVS12+3A>G founder in Inuit/Arctic populations (Canada, Greenland). Gene panel required to distinguish PCCA vs PCCB. Same annual echo, same management, same transplant criteria.',
  'IVD':    'ISOVALERIC ACIDEMIA — C5 (ISOVALERYLCARNITINE) on NBS = PATHOGNOMONIC (exclusive to IVD). SWEATY FEET / CHEESY BODY ODOR from volatile isovaleric acid = immediate clinical clue. A282V MILD VARIANT: most NBS-identified IVA — likely benign; avoid over-treatment. Exclude pivalate drugs (pivampicillin/pivmecillinam) — also cause C5 elevation. Glycine + carnitine are conjugation therapies.',
  'GCDH':   'GLUTARIC ACIDURIA TYPE 1 — MACROCEPHALY (>97th %ile) = should prompt GA1 testing. BAT-WING frontotemporal atrophy on MRI = PATHOGNOMONIC (present before neurological injury). STRIATAL CRISIS after febrile illness during 3-36 months window → permanent acute-onset dystonia. LOW EXCRETOR TRAP: 30-40% have near-normal C5DC and urine GA — molecular testing MANDATORY. EMERGENCY FEVER PROTOCOL saves striatum.',
  'HMGCL':  'HMG-COA LYASE DEFICIENCY — HYPOKETOTIC HYPOGLYCEMIA = PATHOGNOMONIC (ketone bodies CANNOT be made; brain starves of both glucose AND backup fuel). 3-HMG + 3-methylglutaconic acid in urine. C6DC on NBS. FASTING ABSOLUTELY CONTRAINDICATED. Emergency: IV 10% glucose IMMEDIATELY — do not wait for confirmation. DDx MCAD: MCAD has C8 elevation + NO 3-HMG in urine.',
  'ACAT1':  '3-OXOTHIOLASE (T2) DEFICIENCY — EPISODIC SEVERE KETOACIDOSIS with COMPLETELY NORMAL INTERVALS = PATHOGNOMONIC. Crisis pH <7.0 but aggressive IV NaHCO3 + glucose → full recovery. C5-OH + C5:1 on NBS; 2-methyl-3-OH-butyric acid (2M3HBA) in urine during crisis. DDx HMGCL: ACAT1 = HYPERketotic; HMGCL = hypoketotic. EXCELLENT long-term prognosis with crisis prevention.',
};

export default function HereditaryOAAtlasPage() {
  const [activeTab, setActiveTab] = useState('Overview');
  const [overview, setOverview] = useState(null);
  const [breakdown, setBreakdown] = useState(null);
  const [definitions, setDefinitions] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedDef, setExpandedDef] = useState(null);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`${API}/api/${SLUG}/overview`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/breakdown`).then(r => r.json()),
      fetch(`${API}/api/${SLUG}/definitions`).then(r => r.json()),
    ])
      .then(([ov, bk, df]) => { setOverview(ov); setBreakdown(bk); setDefinitions(df); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, []);

  if (loading) return <div className="p-6 text-center text-gray-400">Loading Hereditary-Organic-Acidemia-MMA-PA-IVA-GA1-Atlas…</div>;
  if (error)   return <div className="p-6 text-center text-red-500">Error: {error}</div>;

  const genes = overview?.genes || [];

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-950 to-slate-900 border-b border-indigo-800 px-6 py-5">
        <h1 className="text-2xl font-bold text-indigo-100">🧬 Hereditary Organic Acidemia Atlas</h1>
        <p className="text-indigo-300 text-sm mt-1">
          MMA · PA · IVA · GA1 · HMG-CoA Lyase · 3-Oxothiolase — 8-Gene Complete Reference
        </p>
        <p className="text-indigo-400 text-xs mt-1">
          MUT / MMACHC / PCCA / PCCB / IVD / GCDH / HMGCL / ACAT1 · 320 patients (8×40) · seeds 4150-4157
        </p>
      </div>

      {/* Tabs */}
      <div className="flex border-b border-gray-800 bg-gray-900 px-6">
        {TABS.map(t => (
          <button
            key={t}
            onClick={() => setActiveTab(t)}
            className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === t
                ? 'border-indigo-400 text-indigo-300'
                : 'border-transparent text-gray-400 hover:text-gray-200'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="p-6">

        {/* ── OVERVIEW TAB ── */}
        {activeTab === 'Overview' && overview && (
          <div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-indigo-300">{overview.total_patients}</div>
                <div className="text-xs text-gray-400 mt-1">Total Patients</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-green-300">{genes.length}</div>
                <div className="text-xs text-gray-400 mt-1">Genes</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-amber-300">40</div>
                <div className="text-xs text-gray-400 mt-1">Patients per Gene</div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 text-center">
                <div className="text-3xl font-bold text-red-300">8</div>
                <div className="text-xs text-gray-400 mt-1">Disease Subtypes</div>
              </div>
            </div>

            {/* Key Clinical Flags */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
              {[
                { gene: 'MUT', flag: 'metabolic_stroke_pct', label: 'Metabolic Stroke Risk', color: 'red' },
                { gene: 'MMACHC', flag: 'retinal_degen_pct', label: 'Retinal Degeneration (cblC)', color: 'orange' },
                { gene: 'PCCA', flag: 'cardiomyopathy_pct', label: 'Cardiomyopathy (PA)', color: 'purple' },
                { gene: 'GCDH', flag: 'striatal_necrosis_pct', label: 'Striatal Necrosis (GA1)', color: 'amber' },
              ].map(({ gene, flag, label, color }) => {
                const s = overview.cohort_summary[gene];
                return (
                  <div key={gene} className={`bg-${color}-950 border border-${color}-800 rounded-lg p-3`}>
                    <div className={`text-xl font-bold text-${color}-300`}>{s?.[flag]}%</div>
                    <div className="text-xs text-gray-400">{label}</div>
                    <div className={`text-xs text-${color}-400 font-mono mt-1`}>{gene}</div>
                  </div>
                );
              })}
            </div>

            {/* Gene cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {genes.map(g => {
                const s = overview.cohort_summary[g];
                const info = GENE_INFO[g] || {};
                return (
                  <div key={g} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <div className="flex items-center gap-2 mb-2">
                      <div className="w-3 h-3 rounded-full" style={{ background: GENE_COLORS[g] || '#888' }} />
                      <span className="font-bold text-gray-100">{g}</span>
                      <span className="text-xs text-gray-400 ml-auto">{info.locus} · {info.inh}</span>
                    </div>
                    <div className="text-xs text-gray-300 mb-2 font-medium">{info.full}</div>
                    <div className="text-xs text-gray-400 mb-3 leading-relaxed">{info.note}</div>
                    <div className="text-xs text-indigo-300 mb-2 font-mono bg-gray-900 rounded px-2 py-1 truncate">{s?.nbs_marker?.substring(0,100)}…</div>
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="bg-gray-900 rounded p-1 text-center">
                        <span className="text-green-300 font-bold">{s?.nbs_positive_pct}%</span>
                        <div className="text-gray-500">NBS Pos</div>
                      </div>
                      <div className="bg-gray-900 rounded p-1 text-center">
                        <span className={`font-bold ${s?.cardiomyopathy_pct > 10 ? 'text-red-300' : 'text-gray-300'}`}>{s?.cardiomyopathy_pct}%</span>
                        <div className="text-gray-500">Cardio.</div>
                      </div>
                      <div className="bg-gray-900 rounded p-1 text-center">
                        <span className={`font-bold ${s?.ketoacidosis_pct > 50 ? 'text-amber-300' : 'text-gray-300'}`}>{s?.ketoacidosis_pct}%</span>
                        <div className="text-gray-500">Ketoacid.</div>
                      </div>
                    </div>
                    {/* Special flags */}
                    <div className="flex flex-wrap gap-1 mt-2">
                      {s?.fasting_CI_flag && <span className="text-xs bg-red-900 text-red-200 rounded px-1">FASTING CI</span>}
                      {s?.macrocephaly_flag && <span className="text-xs bg-orange-900 text-orange-200 rounded px-1">MACROCEPHALY</span>}
                      {s?.striatal_necrosis_flag && <span className="text-xs bg-amber-900 text-amber-200 rounded px-1">STRIATAL RISK</span>}
                      {s?.hypoketotic_flag && <span className="text-xs bg-teal-900 text-teal-200 rounded px-1">HYPOKETOTIC</span>}
                      {s?.b12_responsive_flag && <span className="text-xs bg-blue-900 text-blue-200 rounded px-1">B12-RESPONSIVE</span>}
                      {s?.sweaty_odor_flag && <span className="text-xs bg-green-900 text-green-200 rounded px-1">SWEATY ODOR</span>}
                      {s?.retinal_degen_flag && <span className="text-xs bg-purple-900 text-purple-200 rounded px-1">RETINAL DEGEN</span>}
                      {s?.combined_hcu_flag && <span className="text-xs bg-pink-900 text-pink-200 rounded px-1">COMBINED MMA+HCU</span>}
                      {s?.cardiomyopathy_flag && <span className="text-xs bg-red-900 text-red-200 rounded px-1">DCM RISK</span>}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Key facts */}
            {overview.key_facts && (
              <div className="mt-6 space-y-4">
                <h3 className="text-lg font-semibold text-indigo-200">Key Clinical Facts</h3>
                {Object.entries(overview.key_facts).map(([k, v]) => (
                  <div key={k} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <div className="text-xs font-mono text-indigo-400 mb-2">{k.replace(/_/g, ' ').toUpperCase()}</div>
                    <p className="text-xs text-gray-300 leading-relaxed">{v}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── GENE TABLE TAB ── */}
        {activeTab === 'Gene Table' && overview && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="bg-gray-800 text-gray-300">
                  <th className="border border-gray-700 px-3 py-2 text-left">Gene</th>
                  <th className="border border-gray-700 px-3 py-2">Locus</th>
                  <th className="border border-gray-700 px-3 py-2">aa/kDa</th>
                  <th className="border border-gray-700 px-3 py-2">NBS Marker</th>
                  <th className="border border-gray-700 px-3 py-2">NBS%</th>
                  <th className="border border-gray-700 px-3 py-2">Neonatal</th>
                  <th className="border border-gray-700 px-3 py-2">NH3↑</th>
                  <th className="border border-gray-700 px-3 py-2">Cardio%</th>
                  <th className="border border-gray-700 px-3 py-2">Stroke%</th>
                  <th className="border border-gray-700 px-3 py-2">Keto%</th>
                  <th className="border border-gray-700 px-3 py-2">HypoKeto%</th>
                  <th className="border border-gray-700 px-3 py-2">Macrocephaly%</th>
                  <th className="border border-gray-700 px-3 py-2">Striatal%</th>
                  <th className="border border-gray-700 px-3 py-2">Retinal%</th>
                  <th className="border border-gray-700 px-3 py-2 text-left">Key Mutation</th>
                </tr>
              </thead>
              <tbody>
                {genes.map((g, i) => {
                  const s = overview.cohort_summary[g];
                  const p = PROFILES_DISPLAY[g] || {};
                  return (
                    <tr key={g} className={i % 2 === 0 ? 'bg-gray-900' : 'bg-gray-800'}>
                      <td className="border border-gray-700 px-3 py-2">
                        <div className="flex items-center gap-1">
                          <div className="w-2 h-2 rounded-full" style={{ background: GENE_COLORS[g] }} />
                          <span className="font-bold">{g}</span>
                        </div>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center text-gray-300">{s?.locus}</td>
                      <td className="border border-gray-700 px-3 py-2 text-center text-gray-300">{s?.size_aa}</td>
                      <td className="border border-gray-700 px-3 py-2 text-gray-300 max-w-xs truncate">{s?.nbs_marker?.substring(0,60)}…</td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={`font-bold ${s?.nbs_positive_pct > 80 ? 'text-green-300' : 'text-amber-300'}`}>{s?.nbs_positive_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={s?.severe_neonatal_pct > 50 ? 'text-red-300 font-bold' : 'text-gray-400'}>{s?.severe_neonatal_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={s?.hyperammonemia_pct > 40 ? 'text-orange-300 font-bold' : 'text-gray-400'}>{s?.hyperammonemia_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={s?.cardiomyopathy_pct > 15 ? 'text-red-300 font-bold' : 'text-gray-400'}>{s?.cardiomyopathy_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={s?.metabolic_stroke_pct > 15 ? 'text-red-300 font-bold' : 'text-gray-400'}>{s?.metabolic_stroke_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={s?.ketoacidosis_pct > 60 ? 'text-amber-300 font-bold' : 'text-gray-400'}>{s?.ketoacidosis_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={s?.hypoketotic_hypo_pct > 40 ? 'text-teal-300 font-bold' : 'text-gray-400'}>{s?.hypoketotic_hypo_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={s?.macrocephaly_pct > 30 ? 'text-orange-300 font-bold' : 'text-gray-400'}>{s?.macrocephaly_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={s?.striatal_necrosis_pct > 20 ? 'text-amber-300 font-bold' : 'text-gray-400'}>{s?.striatal_necrosis_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-center">
                        <span className={s?.retinal_degen_pct > 30 ? 'text-purple-300 font-bold' : 'text-gray-400'}>{s?.retinal_degen_pct}%</span>
                      </td>
                      <td className="border border-gray-700 px-3 py-2 text-gray-300 max-w-xs">
                        <div className="truncate max-w-xs">{s?.key_mutation?.substring(0,80)}…</div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* ── OA GROUPS & NBS PATHWAYS TAB ── */}
        {activeTab === 'OA Groups & NBS Pathways' && breakdown && (
          <div className="space-y-6">
            {/* Group cards */}
            <h3 className="text-lg font-semibold text-indigo-200">Disease Groups</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(breakdown.groups || {}).map(([gk, gv]) => (
                <div key={gk} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                  <div className="font-bold text-indigo-300 text-sm mb-1">{gk.replace(/_/g,' ').toUpperCase()}</div>
                  <div className="text-xs text-gray-300 mb-2">{gv.mechanism}</div>
                  <div className="flex flex-wrap gap-1 mb-2">
                    {(gv.genes || []).map(g => (
                      <span key={g} className="text-xs px-2 py-0.5 rounded" style={{ background: GENE_COLORS[g] || '#555', color: '#fff' }}>{g}</span>
                    ))}
                  </div>
                  <div className="text-xs text-gray-400">N = {gv.n_patients} patients</div>
                  {Object.entries(gv).filter(([k]) => k.endsWith('_pct')).map(([k, v]) => (
                    <div key={k} className="flex items-center gap-2 mt-1">
                      <div className="text-xs text-gray-400 w-40 truncate">{k.replace(/_pct$/,'').replace(/_/g,' ')}</div>
                      <div className="flex-1 bg-gray-700 rounded-full h-2">
                        <div className="bg-indigo-500 h-2 rounded-full" style={{ width: `${v}%` }} />
                      </div>
                      <div className="text-xs text-gray-300 w-10 text-right">{v}%</div>
                    </div>
                  ))}
                </div>
              ))}
            </div>

            {/* Cascade / NBS algorithm */}
            {definitions?.cascade && (
              <div>
                <h3 className="text-lg font-semibold text-indigo-200 mt-6 mb-4">NBS OA Cascade Algorithm</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(definitions.cascade).map(([pathway, steps]) => (
                    <div key={pathway} className="bg-gray-800 rounded-lg p-4 border border-indigo-800">
                      <div className="font-bold text-amber-300 text-sm mb-2">{pathway.replace(/_/g,' ').toUpperCase()}</div>
                      {Object.entries(steps).map(([step, val]) => (
                        <div key={step} className="text-xs text-gray-300 mb-1">
                          <span className="text-indigo-400 font-mono">{step}: </span>{val}
                        </div>
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Per-gene breakdown bars */}
            <h3 className="text-lg font-semibold text-indigo-200 mt-6">Per-Gene Clinical Metrics</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {genes.map(g => {
                const gb = breakdown.gene_breakdown?.[g];
                if (!gb) return null;
                const metrics = [
                  ['NBS Positive', gb.nbs_positive_pct, 'green'],
                  ['Severe Neonatal', gb.severe_neonatal_pct, 'red'],
                  ['Hyperammonemia', gb.hyperammonemia_pct, 'orange'],
                  ['Cardiomyopathy', gb.cardiomyopathy_pct, 'red'],
                  ['Metabolic Stroke', gb.metabolic_stroke_pct, 'red'],
                  ['Ketoacidosis', gb.ketoacidosis_pct, 'amber'],
                  ['Hypoketotic Hypogly.', gb.hypoketotic_hypo_pct, 'teal'],
                  ['Macrocephaly', gb.macrocephaly_pct, 'orange'],
                  ['Striatal Necrosis', gb.striatal_necrosis_pct, 'amber'],
                  ['Retinal Degeneration', gb.retinal_degen_pct, 'purple'],
                ];
                return (
                  <div key={g} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <div className="flex items-center gap-2 mb-3">
                      <div className="w-3 h-3 rounded-full" style={{ background: GENE_COLORS[g] }} />
                      <span className="font-bold text-sm">{g}</span>
                      <span className="text-xs text-gray-400 ml-auto">onset: {gb.mean_age_onset_m}m · delay: {gb.mean_dx_delay_m}m</span>
                    </div>
                    {metrics.map(([label, val, color]) => val > 0 && (
                      <div key={label} className="flex items-center gap-2 mb-1">
                        <div className="text-xs text-gray-400 w-36 truncate">{label}</div>
                        <div className="flex-1 bg-gray-700 rounded-full h-1.5">
                          <div
                            className={`bg-${color}-500 h-1.5 rounded-full`}
                            style={{ width: `${val}%` }}
                          />
                        </div>
                        <div className={`text-xs text-${color}-300 w-10 text-right font-mono`}>{val}%</div>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* ── DEFINITIONS TAB ── */}
        {activeTab === 'Definitions' && definitions && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-indigo-200">Gene Definitions & Clinical References</h3>
            {genes.map(g => {
              const d = definitions.definitions?.[g];
              const isOpen = expandedDef === g;
              return (
                <div key={g} className="bg-gray-800 rounded-lg border border-gray-700">
                  <button
                    onClick={() => setExpandedDef(isOpen ? null : g)}
                    className="w-full text-left px-5 py-3 flex items-center gap-3"
                  >
                    <div className="w-3 h-3 rounded-full" style={{ background: GENE_COLORS[g] }} />
                    <span className="font-bold text-gray-100">{g}</span>
                    <span className="text-xs text-gray-400 ml-2">{d?.full_name}</span>
                    <span className="ml-auto text-gray-400">{isOpen ? '▲' : '▼'}</span>
                  </button>
                  {isOpen && d && (
                    <div className="px-5 pb-4 border-t border-gray-700 pt-3 space-y-3">
                      {/* Pathognomonic highlight */}
                      <div className="bg-red-950 border border-red-800 rounded p-3">
                        <div className="text-xs font-bold text-red-300 mb-1">PATHOGNOMONIC / KEY CLINICAL</div>
                        <div className="text-xs text-red-200">{PATHOGNOMONIC[g]}</div>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                        <div><span className="text-gray-400">Size: </span><span className="text-gray-200">{d.protein_size}</span></div>
                        <div><span className="text-gray-400">Locus: </span><span className="text-gray-200">{d.locus}</span></div>
                        <div><span className="text-gray-400">OMIM gene: </span><span className="text-gray-200">{d.omim_gene}</span></div>
                        <div><span className="text-gray-400">Disease: </span><span className="text-gray-200">{d.omim_disease}</span></div>
                      </div>
                      <div>
                        <div className="text-xs font-semibold text-indigo-300 mb-1">Mechanism</div>
                        <p className="text-xs text-gray-300 leading-relaxed">{d.mechanism_detail}</p>
                      </div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                        <div>
                          <div className="font-semibold text-green-300 mb-1">Treatment</div>
                          <div className="text-gray-300">{d.treatment}</div>
                        </div>
                        <div>
                          <div className="font-semibold text-blue-300 mb-1">Surveillance</div>
                          <div className="text-gray-300">{d.surveillance}</div>
                        </div>
                        <div>
                          <div className="font-semibold text-amber-300 mb-1">Key DDx</div>
                          <div className="text-gray-300">{d.key_ddx}</div>
                        </div>
                        <div>
                          <div className="font-semibold text-purple-300 mb-1">OMIM</div>
                          <div className="text-gray-300 break-all">{d.omim_link}</div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

// Static display profiles (fallback for Gene Table tab before API loads)
const PROFILES_DISPLAY = {
  'MUT':    { locus: '6p12.3', aa: '750aa/83kDa' },
  'MMACHC': { locus: '1p34.1', aa: '282aa/32kDa' },
  'PCCA':   { locus: '13q32.3', aa: '728aa/80kDa' },
  'PCCB':   { locus: '3q22.3', aa: '539aa/58kDa' },
  'IVD':    { locus: '15q15.1', aa: '394aa/46kDa' },
  'GCDH':   { locus: '19p13.13', aa: '438aa/48kDa' },
  'HMGCL':  { locus: '1p36.11', aa: '325aa/34kDa' },
  'ACAT1':  { locus: '11q22.3', aa: '427aa/45kDa' },
};
