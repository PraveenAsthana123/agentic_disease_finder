"""Integration Dashboard — EMR/FHIR, device streaming, API contracts, webhooks.

Reads: config/integrations.json, config/iot_devices.json,
       config/admin_module.json, config/neurolab_readiness.json
"""

import json
import os
from collections import Counter

_DIR = os.path.dirname(__file__)
_CFG = os.path.join(_DIR, '..', 'config')


def _load(fname):
    path = os.path.join(_CFG, fname)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def overview():
    """Summary KPIs: integration counts, device counts, readiness percentage,
    status distributions by category."""
    ig = _load('integrations.json')
    iot = _load('iot_devices.json')
    am = _load('admin_module.json')

    if not ig or not iot:
        return {"available": False, "note": "Config files missing"}

    integrations = ig.get('integrations', [])
    delivery_channels = ig.get('delivery_channels', [])
    devices = iot.get('devices', [])
    admin_integrations = (am or {}).get('integrations', [])

    # --- KPIs ---
    total_integrations = len(integrations)
    total_devices = len(devices)

    int_status = Counter(i.get('status', '?') for i in integrations)
    integrations_live = int_status.get('live', 0) + int_status.get('built', 0)
    integrations_needs_credentials = int_status.get('needs-credentials', 0)
    integrations_planned = int_status.get('planned', 0)

    dc_total = len(delivery_channels)
    dc_status = Counter(d.get('status', '?') for d in delivery_channels)
    dc_live = dc_status.get('built', 0) + dc_status.get('live', 0)

    dev_status = Counter(d.get('status', '?') for d in devices)
    devices_partial = dev_status.get('partial', 0)
    devices_planned = dev_status.get('planned', 0)

    api_contracts_registered = len(admin_integrations)

    # --- Overall readiness ---
    all_items = integrations + delivery_channels + devices
    total_all = len(all_items)
    live_count = 0
    partial_count = 0
    for item in all_items:
        s = item.get('status', '?')
        if s in ('live', 'built'):
            live_count += 1
        elif s == 'partial':
            partial_count += 1
    overall_readiness_pct = round(
        (live_count + partial_count * 0.5) / total_all * 100, 1
    ) if total_all else 0

    # --- Charts ---
    integration_status_distribution = [
        {"status": s, "count": c}
        for s, c in sorted(int_status.items(), key=lambda x: -x[1])
    ]

    delivery_channel_distribution = [
        {"status": s, "count": c}
        for s, c in sorted(dc_status.items(), key=lambda x: -x[1])
    ]

    device_status_distribution = [
        {"status": s, "count": c}
        for s, c in sorted(dev_status.items(), key=lambda x: -x[1])
    ]

    cat_counts = Counter(i.get('category', 'uncategorized') for i in integrations)
    integration_by_category = [
        {"category": cat, "count": c}
        for cat, c in sorted(cat_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "available": True,
        "summary": {
            "total_integrations": total_integrations,
            "total_devices": total_devices,
            "integrations_live": integrations_live,
            "integrations_needs_credentials": integrations_needs_credentials,
            "integrations_planned": integrations_planned,
            "delivery_channels_total": dc_total,
            "delivery_channels_live": dc_live,
            "devices_partial": devices_partial,
            "devices_planned": devices_planned,
            "api_contracts_registered": api_contracts_registered,
            "overall_readiness_pct": overall_readiness_pct,
        },
        "integration_status_distribution": integration_status_distribution,
        "delivery_channel_distribution": delivery_channel_distribution,
        "device_status_distribution": device_status_distribution,
        "integration_by_category": integration_by_category,
    }


def breakdown():
    """Detailed breakdown: full integration lists, device fleet, admin
    integrations, and stakeholder integration gaps."""
    ig = _load('integrations.json')
    iot = _load('iot_devices.json')
    am = _load('admin_module.json')
    nr = _load('neurolab_readiness.json')

    if not ig or not iot:
        return {"available": False}

    integrations = ig.get('integrations', [])
    delivery_channels = ig.get('delivery_channels', [])
    devices = iot.get('devices', [])
    admin_integrations = (am or {}).get('integrations', [])

    # --- Stakeholder integration gaps ---
    integration_keywords = [
        'EMR', 'FHIR', 'streaming', 'device', 'DICOM', 'mobile',
        'integration', 'sync', 'e-signature', 'HL7',
    ]
    stakeholders = (nr or {}).get('stakeholders', [])
    stakeholder_integration_gaps = []
    for sh in stakeholders:
        role = sh.get('role', '?')
        missing = sh.get('missing', [])
        integration_missing = [
            m for m in missing
            if any(kw.lower() in m.lower() for kw in integration_keywords)
        ]
        if integration_missing:
            stakeholder_integration_gaps.append({
                "role": role,
                "integration_gaps": integration_missing,
                "gap_count": len(integration_missing),
            })

    return {
        "available": True,
        "integrations": integrations,
        "delivery_channels": delivery_channels,
        "devices": devices,
        "admin_integrations": admin_integrations,
        "stakeholder_integration_gaps": stakeholder_integration_gaps,
    }


def definitions():
    """Integration and connectivity terminology."""
    return {
        "available": True,
        "definitions": [
            {"term": "Integration", "definition": "External system connection (EMR/FHIR, cloud storage, messaging) that exchanges data with the platform via APIs, webhooks, or MCP connectors."},
            {"term": "Delivery Channel", "definition": "Method for delivering assessments to patients (voice AI, conversational AI, web form, email link, survey link, email campaign)."},
            {"term": "IoT Device", "definition": "Internet of Things device for patient monitoring (EEG headset, smartwatch, seizure-detection wristband) that streams or buffers clinical data."},
            {"term": "API Contract", "definition": "Defined interface for data exchange between systems, specifying endpoints, methods, request/response schemas, and authentication requirements."},
            {"term": "Webhook", "definition": "Automated HTTP callback triggered by an event (e.g., seizure alert, assessment completion) that notifies an external system in real time."},
            {"term": "EMR/FHIR", "definition": "Electronic Medical Record / Fast Healthcare Interoperability Resources standard for structured clinical data exchange between healthcare systems."},
            {"term": "OAuth", "definition": "Authorization framework for secure third-party access, allowing integrations (Google Drive, Slack, etc.) to act on behalf of users without sharing credentials."},
            {"term": "MCP Connector", "definition": "Model Context Protocol connector for standardized tool integration, providing a uniform interface for AI agents to interact with external services."},
            {"term": "Store-and-Forward", "definition": "Offline data buffering strategy where devices store data locally and synchronize with the backend on reconnect, ensuring zero data loss."},
            {"term": "Edge Model", "definition": "AI model running locally on device for real-time inference (e.g., on-device seizure detection) without requiring cloud connectivity."},
        ],
    }


if __name__ == '__main__':
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
