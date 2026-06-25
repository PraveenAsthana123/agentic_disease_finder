#!/usr/bin/env bash
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
python3 - <<'PY'
import clinical_db as cdb
d=cdb.list_requests()
print("════════ 📥 REQUEST INBOX (your inputs + status) ════════")
print("status:",d["by_status"],"| 🔔 unaddressed:",d["open_count"])
op=[r for r in d["items"] if r["status"] in ("open","pending")]
print("\n⏳ PENDING / UNADDRESSED:" if op else "\n✅ all inputs addressed/closed.")
for r in op: print(f"  #{r['id']} {r['request_text']}")
ad=[r for r in d["items"] if r["status"]=="addressed" and r.get("impl_tab")]
print("\n✅ ADDRESSED (where implemented):")
for r in ad[:14]:
    print(f"  #{r['id']} {r['request_text'][:50]}")
    print(f"       tab={r.get('impl_tab')} · api={r.get('impl_api')} · tested={r.get('tested')}")
ni=[r for r in d["items"] if r["status"] in ("not-implemented","rejected")]
if ni:
    print("\n🚫 NOT-IMPLEMENTED / REJECTED:")
    for r in ni: print(f"  #{r['id']} [{r['status']}] {r['request_text']}")
PY
