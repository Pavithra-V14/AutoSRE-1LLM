from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

NOW = datetime.now()


def _ts(minutes_ago: int) -> str:
    return str(NOW - timedelta(minutes=minutes_ago))


SCENARIOS: dict[str, dict[str, Any]] = {

    # -- A. Deploy-caused failures ------------------------------------------

    "db_timeout": {
        "description": "A config change reduced DB connection pool size, "
        "causing timeouts under normal load.",
        "ground_truth_cause": "deploy:cfg-4471 -> pool_size=10 -> connection_exhaustion -> latency_spike",
        "expected_tools": ["get_deploy_history", "get_metrics", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "checkout-service": {
                "p99_latency_ms": [120, 130, 125, 480, 510, 495],
                "error_rate": [0.001, 0.001, 0.002, 0.08, 0.09, 0.085],
            }
        },
        "logs": [
            {"ts": _ts(8), "level": "ERROR", "message": "connection pool exhausted, waiting for available connection"},
            {"ts": _ts(7), "level": "ERROR", "message": "connection pool exhausted, waiting for available connection"},
        ],
        "deploys": [
            {"id": "cfg-4471", "ts": _ts(12), "service": "checkout-service", "change": "db.pool_size: 50 -> 10"},
        ],
    },

    "bad_deploy_rollback": {
        "description": "A bad code deploy introduced a null-pointer bug on a hot path.",
        "ground_truth_cause": "deploy:rel-892 -> null_check_removed -> crash_loop",
        "expected_tools": ["get_deploy_history", "get_metrics", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "inventory-service": {
                "p99_latency_ms": [90, 95, 92, 90, 88],
                "error_rate": [0.001, 0.001, 0.31, 0.42, 0.39],
            }
        },
        "logs": [
            {"ts": _ts(5), "level": "FATAL", "message": "NullPointerException at InventoryHandler.process"},
        ],
        "deploys": [
            {"id": "rel-892", "ts": _ts(6), "service": "inventory-service", "change": "remove redundant null check (PR #4021)"},
        ],
    },

    "memory_leak_deploy": {
        "description": "A deploy introduced unbounded in-memory cache growth, "
        "leading to OOM kills under sustained traffic.",
        "ground_truth_cause": "deploy:rel-1103 -> unbounded_cache_growth -> oom_kill -> pod_restarts",
        "expected_tools": ["get_deploy_history", "get_metrics", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "recommendation-service": {
                "memory_used_mb": [512, 600, 1400, 1900, 2040, 2048],
                "restarts": [0, 0, 0, 1, 2, 3],
            }
        },
        "logs": [
            {"ts": _ts(9), "level": "WARN", "message": "memory usage at 95% of container limit"},
            {"ts": _ts(3), "level": "FATAL", "message": "OOMKilled: container exceeded memory limit"},
        ],
        "deploys": [
            {"id": "rel-1103", "ts": _ts(40), "service": "recommendation-service", "change": "add response cache, no TTL/eviction configured"},
        ],
    },

    "feature_flag_deploy": {
        "description": "A feature flag flip increased calls to a downstream "
        "pricing API beyond its rate limit.",
        "ground_truth_cause": "deploy:flag-dynamic_pricing_v2 -> pricing_api_call_volume_up -> rate_limit_exceeded -> checkout_errors",
        "expected_tools": ["get_deploy_history", "get_metrics", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "pricing-client": {
                "requests_per_sec": [40, 42, 210, 230, 225],
                "error_rate": [0.0, 0.0, 0.22, 0.28, 0.25],
            }
        },
        "logs": [
            {"ts": _ts(6), "level": "ERROR", "message": "pricing-api responded 429 Too Many Requests"},
        ],
        "deploys": [
            {"id": "flag-dynamic_pricing_v2", "ts": _ts(15), "service": "checkout-service", "change": "enable dynamic_pricing_v2 for 100% of traffic"},
        ],
    },

    # -- B. Traffic / scaling issues -----------------------------------------

    "traffic_spike_no_deploy": {
        "description": "Organic traffic spike with no recent deploy -- the "
        "agent should NOT blame a deploy here (negative test case).",
        "ground_truth_cause": "traffic_spike -> autoscaling_lag -> latency_spike",
        "expected_tools": ["get_metrics", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "search-service": {
                "p99_latency_ms": [100, 105, 310, 340, 300],
                "requests_per_sec": [200, 210, 850, 900, 870],
            }
        },
        "logs": [
            {"ts": _ts(4), "level": "WARN", "message": "autoscaler adding 3 replicas, target reached in ~90s"},
        ],
        "deploys": [],  # deliberately empty -- tests fail-closed / no-fabrication behavior
    },

    "flash_sale_traffic": {
        "description": "A scheduled marketing flash sale drove a legitimate "
        "10x traffic surge with no code or config change involved.",
        "ground_truth_cause": "marketing_event -> traffic_surge -> autoscaling_lag -> checkout_latency",
        "expected_tools": ["get_metrics"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "checkout-service": {
                "p99_latency_ms": [110, 115, 520, 610, 480, 300],
                "requests_per_sec": [150, 160, 1500, 1800, 1600, 1400],
            }
        },
        "logs": [
            {"ts": _ts(20), "level": "INFO", "message": "flash sale campaign 'summer_drop' went live"},
            {"ts": _ts(15), "level": "WARN", "message": "HPA scaling checkout-service from 6 to 24 replicas"},
        ],
        "deploys": [],
    },

    "bot_traffic_spike": {
        "description": "Anomalous bot/scraper traffic saturated the rate "
        "limiter, degrading service for real users. No deploy involved.",
        "ground_truth_cause": "bot_traffic -> rate_limiter_saturation -> legitimate_request_throttling",
        "expected_tools": ["get_metrics"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "product-catalog-service": {
                "requests_per_sec": [300, 310, 4200, 4500, 4100],
                "error_rate": [0.001, 0.001, 0.18, 0.21, 0.19],
            }
        },
        "logs": [
            {"ts": _ts(10), "level": "WARN", "message": "rate limiter triggered for 40+ distinct IPs, pattern suggests scraping"},
        ],
        "deploys": [],
    },

    # -- C. Cascading downstream failures ------------------------------------

    "upstream_dependency_outage": {
        "description": "A third-party payment gateway outage cascades into "
        "checkout-service timeouts.",
        "ground_truth_cause": "payment_gateway_outage -> upstream_timeout -> checkout_service_errors",
        "expected_tools": ["get_metrics", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "checkout-service": {
                "p99_latency_ms": [130, 135, 5000, 5000, 4800],
                "error_rate": [0.001, 0.002, 0.55, 0.61, 0.58],
            }
        },
        "logs": [
            {"ts": _ts(7), "level": "ERROR", "message": "timeout calling payment-gateway.external: 5000ms exceeded"},
            {"ts": _ts(6), "level": "ERROR", "message": "payment-gateway status page reports partial outage"},
        ],
        "deploys": [],
    },

    "cache_invalidation_storm": {
        "description": "A deploy changed cache eviction policy, causing a "
        "thundering-herd of requests to hit the database simultaneously.",
        "ground_truth_cause": "deploy:cfg-5502 -> aggressive_cache_eviction -> thundering_herd -> db_load_spike",
        "expected_tools": ["get_deploy_history", "get_metrics"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "product-db": {
                "cpu_percent": [35, 38, 92, 97, 95],
                "active_connections": [40, 42, 480, 500, 495],
            }
        },
        "logs": [
            {"ts": _ts(9), "level": "WARN", "message": "cache hit rate dropped from 92% to 11%"},
            {"ts": _ts(8), "level": "ERROR", "message": "database connection pool near saturation"},
        ],
        "deploys": [
            {"id": "cfg-5502", "ts": _ts(14), "service": "product-catalog-service", "change": "cache.ttl_seconds: 3600 -> 30"},
        ],
    },

    "cascading_retry_storm": {
        "description": "A slowdown in one downstream service triggers retry "
        "amplification that overwhelms two other services.",
        "ground_truth_cause": "inventory_service_slowdown -> retry_amplification -> order_service_overload -> checkout_service_errors",
        "expected_tools": ["get_metrics", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "order-service": {
                "requests_per_sec": [200, 210, 1800, 2100, 1900],
                "error_rate": [0.001, 0.002, 0.32, 0.38, 0.35],
            }
        },
        "logs": [
            {"ts": _ts(11), "level": "WARN", "message": "inventory-service p99 latency up 8x"},
            {"ts": _ts(9), "level": "ERROR", "message": "order-service retry budget exhausted, circuit breaker not configured"},
        ],
        "deploys": [],
    },

    # -- D. Negative / no real incident ---------------------------------------

    "flaky_test_false_alarm": {
        "description": "A brief metric blip self-resolved within 2 minutes "
        "with no user-facing impact -- the agent should conclude low "
        "confidence rather than fabricate a strong causal claim.",
        "ground_truth_cause": "transient_blip -> self_resolved -> no_action_needed",
        "expected_tools": ["get_metrics"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "auth-service": {
                "p99_latency_ms": [95, 98, 340, 100, 97],
                "error_rate": [0.001, 0.001, 0.02, 0.001, 0.001],
            }
        },
        "logs": [],
        "deploys": [],
        "expects_low_confidence": True,
    },

    "scheduled_maintenance_window": {
        "description": "Elevated latency during a pre-announced maintenance "
        "window -- expected behavior, not an incident.",
        "ground_truth_cause": "scheduled_maintenance -> expected_latency -> no_incident",
        "expected_tools": ["get_metrics"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "search-service": {
                "p99_latency_ms": [110, 115, 400, 410, 405],
            }
        },
        "logs": [
            {"ts": _ts(30), "level": "INFO", "message": "scheduled maintenance window started: search-index reindex"},
        ],
        "deploys": [],
        "expects_low_confidence": True,
    },

    # -- E. PREFERRED tool degraded (fail-open expected) ----------------------

    "db_timeout_degraded_logs": {
        "description": "Same underlying cause as db_timeout, but the log "
        "aggregator is returning empty results for this incident -- the "
        "agent should still reach the right conclusion from metrics + "
        "deploy history, disclosing the missing log signal.",
        "ground_truth_cause": "deploy:cfg-4471 -> pool_size=10 -> connection_exhaustion -> latency_spike",
        "expected_tools": ["get_deploy_history", "get_metrics", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "checkout-service": {
                "p99_latency_ms": [120, 130, 125, 480, 510, 495],
                "error_rate": [0.001, 0.001, 0.002, 0.08, 0.09, 0.085],
            }
        },
        "logs": [],  # would normally have entries -- degraded_tools makes this moot
        "deploys": [
            {"id": "cfg-4471", "ts": _ts(12), "service": "checkout-service", "change": "db.pool_size: 50 -> 10"},
        ],
        "degraded_tools": ["get_logs"],
    },

    "deploy_rollback_missing_metrics": {
        "description": "Same underlying cause as bad_deploy_rollback, but the "
        "metrics backend is degraded -- the agent should still identify the "
        "cause from logs + deploy history alone.",
        "ground_truth_cause": "deploy:rel-892 -> null_check_removed -> crash_loop",
        "expected_tools": ["get_deploy_history", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {},
        "logs": [
            {"ts": _ts(5), "level": "FATAL", "message": "NullPointerException at InventoryHandler.process"},
        ],
        "deploys": [
            {"id": "rel-892", "ts": _ts(6), "service": "inventory-service", "change": "remove redundant null check (PR #4021)"},
        ],
        "degraded_tools": ["get_metrics"],
    },

    # -- F. REQUIRED tool degraded (fail-closed expected) ----------------------

    "deploy_correlation_unreachable": {
        "description": "Metrics and logs point clearly at a deploy-shaped "
        "failure, but deploy-history is unreachable -- the agent should "
        "explicitly state it cannot confirm the deploy correlation rather "
        "than guessing.",
        "ground_truth_cause": "deploy_history_unreachable -> cannot_confirm_deploy_correlation",
        "expected_tools": ["get_metrics", "get_logs", "get_deploy_history"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "checkout-service": {
                "p99_latency_ms": [120, 130, 125, 480, 510, 495],
                "error_rate": [0.001, 0.001, 0.002, 0.08, 0.09, 0.085],
            }
        },
        "logs": [
            {"ts": _ts(8), "level": "ERROR", "message": "connection pool exhausted, waiting for available connection"},
        ],
        "deploys": [
            {"id": "cfg-4471", "ts": _ts(12), "service": "checkout-service", "change": "db.pool_size: 50 -> 10"},
        ],
        "degraded_tools": ["get_deploy_history"],
        "expects_fail_closed": True,
    },

    "metrics_unreachable_ambiguous": {
        "description": "Logs and deploy history are available but metrics "
        "are unreachable -- if metrics are marked REQUIRED for this "
        "investigation, the agent should disclose the gap rather than "
        "assert a confident latency/error-rate claim it can't back up.",
        "ground_truth_cause": "metrics_unreachable -> cannot_confirm_severity",
        "expected_tools": ["get_logs", "get_deploy_history"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {},
        "logs": [
            {"ts": _ts(8), "level": "ERROR", "message": "connection pool exhausted, waiting for available connection"},
        ],
        "deploys": [
            {"id": "cfg-4471", "ts": _ts(12), "service": "checkout-service", "change": "db.pool_size: 50 -> 10"},
        ],
        "degraded_tools": ["get_metrics"],
        "expects_fail_closed": True,
    },

    # -- G. Misc diversity ------------------------------------------------------

    "database_deadlock": {
        "description": "A schema migration deploy introduced a new index "
        "causing row-lock contention and deadlocks under concurrent writes.",
        "ground_truth_cause": "deploy:mig-2291 -> new_index_lock_contention -> deadlocks -> write_failures",
        "expected_tools": ["get_deploy_history", "get_metrics", "get_logs"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "orders-db": {
                "deadlocks_per_min": [0, 0, 12, 15, 14],
                "write_latency_ms": [8, 9, 240, 280, 260],
            }
        },
        "logs": [
            {"ts": _ts(7), "level": "ERROR", "message": "deadlock detected on orders table, transaction rolled back"},
        ],
        "deploys": [
            {"id": "mig-2291", "ts": _ts(18), "service": "orders-db", "change": "add composite index on (customer_id, status)"},
        ],
    },

    "resource_quota_exceeded": {
        "description": "A deploy increased replica count beyond the "
        "namespace's resource quota, causing new pods to stay pending.",
        "ground_truth_cause": "deploy:rel-1288 -> replica_count_increase -> quota_exceeded -> pods_pending",
        "expected_tools": ["get_deploy_history", "get_metrics"],  # what a competent investigation should call, for tool-selection accuracy
        "metrics": {
            "checkout-service": {
                "pods_running": [6, 6, 6, 6, 6],
                "pods_pending": [0, 0, 6, 6, 6],
            }
        },
        "logs": [
            {"ts": _ts(4), "level": "ERROR", "message": "FailedScheduling: exceeded quota: cpu limit reached"},
        ],
        "deploys": [
            {"id": "rel-1288", "ts": _ts(6), "service": "checkout-service", "change": "replicas: 6 -> 12"},
        ],
    },
}
