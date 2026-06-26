# Slack Integration

Posts project events to Slack via an Incoming Webhook: watchdog server DOWN/
RECOVERED, Claude→Ollama failover, and (extendable) build/health alerts.

## One-time setup (operator)
1. Create a Slack Incoming Webhook → https://api.slack.com/messaging/webhooks
2. Save the URL one of two ways:
   ```bash
   mkdir -p ~/.config/agenticfinder
   echo 'https://hooks.slack.com/services/XXX/YYY/ZZZ' > ~/.config/agenticfinder/slack_webhook
   chmod 600 ~/.config/agenticfinder/slack_webhook
   # OR: export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/XXX/YYY/ZZZ
   ```
3. Test:
   ```bash
   bash scripts/slack_notify.sh --check   # config status
   bash scripts/slack_notify.sh --test    # send a test message
   ```

## What's wired
| Event | Level | Source |
|---|---|---|
| Server DOWN (auto-restarting) | 🔴 error | scripts/watchdog.sh |
| Server RECOVERED | 🟠 warn | scripts/watchdog.sh |
| Claude limit → Ollama failover | 🟠 warn | scripts/claude_limit_detector.sh |

Add more: `bash scripts/slack_notify.sh --level error "your message"` anywhere.

## Safe by design
- Graceful no-op when not configured (never crashes the watchdog/health callers).
- Every send logged to `jobs/logs/slack.jsonl`. Webhook never committed (gitignored config).
