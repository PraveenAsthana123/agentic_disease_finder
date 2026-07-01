# Pending Tasks — Epilepsy EEG DBA Platform

_Generated 2026-06-23. Honest status. Scope: epilepsy-first._

## ✅ Done & verified
- [x] EEG analysis pipeline (parse EDF/BDF/CSV → 47 features → trained model → signal analysis) — verified on real CHB-MIT EDF
- [x] Per-disease models loaded + multi-disease accuracy job
- [x] Subject-wise (leakage-free) accuracy job — the defensible number
- [x] Multi-agent (Mixture-of-Agents) + MCP analysis job — A2A + MCP healthy
- [x] Clinical DB: patients + 13 capture tables + seizure_metadata + transaction_log + patient_master
- [x] **Transaction history** — UTC + Calgary-local timestamp per write; `/api/transactions` GET+POST; Consultants→Transactions sub-tab with live sign-off
- [x] **Place to upload docs** — (a) Departments→Data tab (EEG analysis), (b) **Patient Master Data** module: multi-format upload (video/pdf/image/.dat/.txt/.docx/.edf) → per-patient folder → master_data.json
- [x] **Data load + folders** — `clinical_data/` drop zones (raw_eeg, video_eeg, realtime, open_science, synthetic, + 13 category folders) with README
- [x] Synthetic epilepsy dataset (30 patients, EEG + video-EEG, all sections, seizure events)
- [x] Multi-format extraction (OCR, PDF, DOCX, EDF, video frames/motion) — verified
- [x] Consultant engagement matrix (10 roles) + per-role 9 sub-tabs + flowchart simulation + sign-offs
- [x] Registries: agent_tasks, dataset_coverage (18 modalities/4 streams), consultant_matrix, **ai_type_coverage (201 from insur_project)**
- [x] Department + governance Main Menu (12 depts) + 2-menu layout

## 🟡 Scaffold (partial — compute/enforcement missing)
- [ ] **Explainable AI** — ground-truth capture built; **SHAP/Grad-CAM compute** not done
- [ ] **Computer vision** — video frame + motion extraction built; classification/segmentation not done
- [ ] **Multimodal fusion** — multi-format ingest built; fusion model not done
- [ ] RAG / GraphRAG — analysis modules present; live pipeline not wired
- [ ] Responsible AI — HITL + governance present; fairness gates not enforced

## ⏳ Pending (named, not built) — pick next
### Models / training
- [ ] **Training pipeline** — train/retrain disease models with subject-wise split + save model card (currently models are pre-trained joblibs)
- [ ] **Deep learning** — CNN / EEGNet / LSTM / Transformer on raw EEG
- [ ] **Conversion models** — audio conversion (ffmpeg+librosa), video transcode/normalize
### Advanced computer vision (video-EEG)
- [ ] **Noise cleaning** — MNE-ICA artifact removal (highest-leverage; improves every model)
- [ ] **Segmentation** — EEG trace digitization from scanned images (U-Net)
- [ ] **Detection** — patient/limb detection in video frames (YOLO)
- [ ] **Classification** — seizure-type / body-movement classification (3D-CNN / pose)
### Data infrastructure
- [ ] **Feature extraction job scheduling** — cron to batch-extract 47 features from new EEG drops
- [ ] **Vector DB + ingest scheduling** — embed clinical text/reports → Chroma/Qdrant + scheduled ingest job
- [ ] **Graph DB + RDFS** — neurophysiology knowledge graph (RDF/RDFS/SPARQL) per §123/§124
### Surfaces
- [ ] **Patient self-service portal** — patient sees ONLY their own data (auth/role-scoped, read-only)
- [ ] Bulk CSV importer — load synthetic/drop-zone CSVs → clinical DB

## ❌ Not-pulled (158 AI types from insur_project)
Domain-irrelevant to epilepsy (banking, oil-and-gas, insurance, drug-discovery, etc.) — see `docs/AI_TYPE_COVERAGE.md`.
