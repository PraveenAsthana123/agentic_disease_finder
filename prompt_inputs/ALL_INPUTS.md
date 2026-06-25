- [ ] **#24** [2026-06-25 09:45:17 MDT] Autonomous build (§159), system unattended. Do ONE iteration only:
1. python3 scripts/next_pending.py — pick the TOP buildable pending item. SKIP: ictal/interictal retrain (too heavy), and anything needing operator credentials/decisions (Gmail/Slack/Drive/auth/EMR/FHIR).
2. Build it for real (backend endpoint + frontend panel + nav wiring), honest §57.7 (no stubs).
3. bash scripts/restart_backend.sh — restart+verify; ONLY proceed if it exits 0 (health 200).
4. Verify the new endpoint returns 200.
5. Mark the registry item built, refresh STATUS.md (python3 scripts/status_report.py).
6. Commit (§51 substrate, §54 NO Co-Authored-By trailer).
7. bash scripts/safe_push.sh (auto-push, fast-forward only).
8. bash scripts/track.sh "built+pushed: <item name>" "autobuild"
If you cannot complete + verify, do NOT commit; run scripts/track.sh with the failure reason and exit. NEVER force-push, NEVER first-publish, NEVER fabricate data, NEVER fake done.
- [ ] **#25** [2026-06-25 09:55:03 MDT] how do I know plan  got created or not , cron job got ceated or not , how many job going to run, which system , are all independing, sequence, if system crash all will stop or still run , will that complete the complete, will that reflect ion UI
- [ ] **#26** [2026-06-25 10:05:52 MDT] have one agent allocated who can file list of issue which I a mnot aware off and guiding
