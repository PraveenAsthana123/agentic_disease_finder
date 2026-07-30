'use client';
import Link from 'next/link';
import { ROLES, FEATURES, AGENTIC } from '../../lib/nav';
import { OPS } from '../../lib/ops';
export default function SiteNav() {
  const dd = (title, items, base) => (
    <li className="nav-item dropdown">
      <a className="nav-link dropdown-toggle text-white" href="#" data-bs-toggle="dropdown">{title}</a>
      <ul className="dropdown-menu dropdown-menu-dark">
        {items.map(i => <li key={i.id}><Link className="dropdown-item" href={`${base}/${i.id}`}>{i.icon?i.icon+' ':''}{i.name}</Link></li>)}
      </ul>
    </li>);
  return (
    <nav className="navbar navbar-expand-lg navbar-dark" style={{background:'#0b1f3a'}}>
      <div className="container-fluid">
        <Link className="navbar-brand fw-bold" href="/">🧠 NeuroAI</Link>
        <ul className="navbar-nav me-auto">
          <li className="nav-item"><Link className="nav-link text-white" href="/dashboard">Dashboard</Link></li>
          {dd('Departments', ROLES, '/role')}
          {dd('Features', FEATURES, '')}
          {dd('Agentic AI', AGENTIC, '/agentic')}
          {dd('Ops & Governance', OPS, '/ops')}
          <li className="nav-item"><Link className="nav-link text-white" href="/drift">📉 Drift</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cognition-link">🔗 Cognition</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-scales">📋 Scales</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-types">AI Types</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/epilepsy-nurse">💉 Epilepsy Nurse</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/medication">&#x1f48a; Medication</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/medication-refills">&#x1f504; Refills</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-severity">&#x26a1; Seizure Severity</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mri-review">&#x1f9e0; MRI Review</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ilae-classification">&#x1f9ec; ILAE Classification</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/incident-management">&#x1f6a8; Incidents</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/inbox">&#x2709;&#xfe0f; Inbox</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/token-cost">&#x1f4b0; Token/Cost</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/vector-db">&#x1f5c4;&#xfe0f; Vector DB</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/knowledge-graph">&#x1f578;&#xfe0f; Knowledge Graph</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-risk">&#x26a0;&#xfe0f; AI Risk</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-governance">&#x1f3db;&#xfe0f; AI Governance</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/workflow">Workflow</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/mcp-security">MCP Security</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/sec-ops">&#x1f512; SecOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/shadow-ai">&#x1f575;&#xfe0f; Shadow AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/change-management">&#x1f504; Change Mgmt</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-retirement">&#x1f4e6; Model Retire</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ai-finops">&#x1f4b8; AI FinOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/billing-claims">&#x1f4b3; Billing &amp; Claims</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/ssep">SSEP</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/vep">VEP</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hrv">HRV</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/abpm">ABPM/Holter</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cloud-ops">&#x2601;&#xfe0f; Cloud Ops</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/data-ops">&#x1f4e6; DataOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/observability">&#x1f441;&#xfe0f; Observability</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/llmops">&#x1f4ac; LLMOps</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/stack">Tech Stack</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/classify">Classify</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/xai">XAI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/interpretable-ai">Interpretable AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/decision-ai">Decision AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/recovery-trajectory">&#x1f4c8; Recovery Trajectory</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/reporting-ai">&#x1f4cb; Reporting AI</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/video-eeg">&#x1f4f9; Video EEG</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/safety-network">&#x1f6e1;&#xfe0f; Safety Network</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/camera-monitoring">&#x1f4f9; Camera Monitor</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pnes-screening">&#x1f9e0; PNES Screening</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/model-comparison">&#x1f4ca; Model Compare</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/hospitalization">&#x1f3e5; Hospitalization</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/seizure-diary">&#x1f4d3; Seizure Diary</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/scheduled-jobs">&#x23f0; Scheduled Jobs</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/onboarding">&#x1f4cb; Onboarding</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/pharmacogenomics">&#x1f9ec; Pharmacogenomics</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/cognitive-decline">&#x1f9e0; Cognitive Decline</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/adl">&#x1f3c3; ADL</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/quality-of-life">&#x1f31f; Quality of Life</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/consent-management">&#x1f4dc; Consent</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/clinical-decisions">&#x2696;&#xfe0f; Clinical Decisions</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/workbench">&#x1f9ea; Workbench</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/neurolab-readiness">&#x1f3e5; NeuroLab Readiness</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/iot-devices">&#x1f4e1; IoT Devices</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/therapy">&#x1f9d8; Therapy</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/multimodal-fusion">&#x1f52c; Multimodal Fusion</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/patient-appointments">&#x1f4c5; Appointments</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/surgical-outcomes">&#x1fa7a; Surgical Outcomes</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/caregiver-readiness">&#x1f91d; Caregiver Readiness</Link></li>
          <li className="nav-item"><Link className="nav-link text-white" href="/r-ipo">&#x1f504; Role IPO</Link></li>
        </ul>
        <span className="navbar-text text-info small">● System Online · SSR</span>
      </div>
    </nav>);
}
