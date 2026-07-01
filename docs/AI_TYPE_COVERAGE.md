# AI-Type Coverage — insur_project (201) vs agenticfinder (epilepsy)

_Generated 2026-07-01T11:21:00-06:00_

**Totals:** built 13 · scaffold 13 · planned 17 · not-pulled 158 (of 201)

## Built (13)
| AI Type | Note |
|---|---|
| agentic-ai | Mixture-of-Agents ensemble (test_agentic_mcp.py) |
| anomaly-detection-ai | IQR+z-score anomaly detection on 47 EEG features (scripts/anomaly_detection_dashboard.py, 3 API endpoints, AnomalyDetectionDashboard.jsx) |
| audit-ai | Transaction log (UTC+local) on every write |
| clinical-ai | Clinical capture tables + per-patient analysis |
| diagnostic-ai | Epilepsy diagnosis prediction + confidence |
| document-ai | PDF/DOCX/text extraction (eeg_ingest) |
| eeg-ai | EEG pipeline: parse + 47 features + analysis (eeg_analysis_pipeline.py) |
| healthcare-ai | EEG epilepsy clinical workflow |
| machine-learning | RandomForest models trained |
| multi-agent-systems | 7 disease agents + orchestrator (MCP) |
| ocr-ai | pytesseract OCR in ingest (eeg_ingest.extract_image) |
| predictive-ai | Per-disease classifiers (models/*.joblib) |
| supervised-learning | Labeled disease classification |

## Scaffold (13)
| AI Type | Note |
|---|---|
| agentic-rag | Analysis module present; not wired to a live corpus |
| ai-control-tower | Department reports + transaction log |
| ai-observability | Transaction log + audit reports |
| computer-vision-ai | Video frame + motion extraction; classification planned |
| explainable-ai | Ground-truth capture table; SHAP/Grad-CAM compute planned |
| governance-ai | Governance dept + consultant oversight registry |
| graphrag | Analysis module present; graph not built |
| human-evaluation-ai | HITL accept/override + reason codes |
| model-governance | Consultant matrix + HITL sign-off |
| model-monitoring-ai | Department report KPIs; drift monitor planned |
| multimodal-ai | Multi-format ingest (video/pdf/img/edf); fusion model planned |
| rag | agentic_rag_analysis/ modules present; live RAG pipeline not wired |
| responsible-ai | HITL + governance reports; fairness gates not enforced |

## Planned (17)
| AI Type | Note |
|---|---|
| bias-detection-ai | Bias assessment — registry |
| causal-ai | Causal seizure analysis — registry |
| deep-learning | CNN/LSTM/Transformer on EEG — agent registry |
| digital-twin-ai | Patient digital twin — registry |
| drift-detection-ai | Model drift monitoring — registry |
| fairness-ai | Fairness gates — registry |
| federated-learning | Multi-site federation — registry |
| foundation-models | Clinical foundation model — registry |
| image-segmentation-ai | EEG trace digitization from images — registry |
| knowledge-graph-ai | Neurophysiology KG — registry |
| llm | LLM-backed explanation — registry |
| neuro-ai | Neuro-specific modeling — registry |
| object-detection-ai | Body-movement detection in video — registry |
| speech-ai | Audio conversion from video-EEG — registry |
| text-to-audio-ai | Audio conversion — registry |
| text-to-video-ai | Video conversion — registry |
| voice-ai | Audio markers — registry |

## Not-pulled (158)
_Available in insur_project, not applicable to the epilepsy DBA scope:_

active-learning, adaptive-rag, adversarial-ai, agent-governance, agentops, agent-swarm-ai, agi, ai-alignment, ai-guardrails, aiops, ai-red-teaming, ai-safety, ai-workforce, aml-ai, ar-ai, asi, autonomous-ai, autonomous-enterprise-ai, autonomous-vehicle-ai, banking-ai, benchmarking-ai, bioinformatics-ai, business-rule-ai, chain-of-thought-ai, climate-ai, cloud-ai, cognitive-ai, collaborative-filtering-ai, compliance-ai, contact-center-ai, content-based-recommendation-ai, contract-ai, conversational-ai, corrective-rag, customer-service-ai, cybersecurity-ai, data-governance-ai, data-observability-ai, dataops-ai, data-quality-ai, decision-intelligence, defect-detection-ai, demand-forecasting-ai, descriptive-ai, distributed-ai, drone-ai, drug-discovery-ai, edge-ai, email-ai, embodied-ai, energy-ai, entity-extraction-ai, ethical-ai, evaluation-ai, face-recognition-ai, finance-ai, financial-forecasting-ai, finops-ai, forecasting-ai, fraud-detection-ai, generative-ai, genomics-ai, geospatial-ai, graph-of-thought-ai, hpc-ai, hr-ai, humanoid-ai, hybrid-recommendation-ai, hybrid-search-ai, image-classification-ai, industrial-robotics-ai, information-retrieval-ai, insurance-ai, intrusion-detection-ai, inventory-ai, iot-ai, legal-ai, limited-memory-ai, lineage-ai, llmops, manufacturing-ai, marketing-ai, materials-ai, medical-imaging-ai, meeting-ai, metadata-ai, mlops, mr-ai, multimodal-rag, neuro-symbolic-ai, nlg, nlp-ai, nlu, odysseus-ai, oil-and-gas-ai, online-learning, ontology-ai, optimization-ai, personalization-ai, physical-ai, pii-detection-ai, planning-ai, policy-ai, prescriptive-ai, pricing-ai, privacy-ai, procurement-ai, quantum-machine-learning, question-answering-ai, ragops, react-ai, reactive-ai, reasoning-ai, recommendation-ai, recruitment-ai, reflection-ai, reinforcement-learning, retail-ai, risk-scoring-ai, rlhf, robotics-ai, route-optimization-ai, sales-ai, sales-forecasting-ai, satellite-ai, schedule-optimization-ai, scientific-ai, self-rag, self-supervised-learning, semantic-ai, semantic-search-ai, semi-supervised-learning, sentiment-ai, simulation-ai, slm, spatial-ai, sql-rag, summarization-ai, supply-chain-ai, symbolic-ai, synthetic-data-ai, taxonomy-ai, text-to-image-ai, theory-of-mind-ai, threat-detection-ai, time-series-ai, tinyml, transfer-learning, translation-ai, tree-of-thought-ai, unsupervised-learning, vector-search-ai, video-analytics-ai, vision-language-models, vr-ai, what-if-ai, world-models, xr-ai
