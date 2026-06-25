#!/usr/bin/env bash
# Conditional auto-push (§159) — pushes ONLY when repo is already shared + fast-forward.
# Refuses: no remote · no upstream · remote branch absent (first-publish) · force needed.
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 2
br=$(git branch --show-current)
git remote get-url origin >/dev/null 2>&1 || { echo "SKIP: no origin remote (not shared)"; exit 3; }
# remote branch must already exist (repo already shared/pushed before)
git ls-remote --exit-code --heads origin "$br" >/dev/null 2>&1 || { echo "SKIP: remote branch '$br' absent — first-publish is operator-gated"; exit 3; }
ahead=$(git rev-list --count origin/"$br".."$br" 2>/dev/null || echo 0)
[ "$ahead" = "0" ] && { echo "nothing to push"; exit 0; }
# fast-forward check: local must contain remote tip (no force needed)
git merge-base --is-ancestor origin/"$br" "$br" 2>/dev/null || { echo "SKIP: not fast-forward — operator-gated (would need force)"; exit 3; }
bash scripts/track.sh "auto-push $ahead commits" "git"; echo "auto-push: $ahead commit(s) → origin/$br (fast-forward)"
git push origin "$br" 2>&1 | tail -2
