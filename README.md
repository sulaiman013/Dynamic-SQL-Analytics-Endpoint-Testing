# Dynamic SQL Analytics Endpoint Schema Refresh in Microsoft Fabric

Programmatically force SQL analytics endpoint metadata sync after Delta table schema changes in Microsoft Fabric, eliminating the manual "Refresh Schema" click.

## The Problem

When you modify Delta tables in a Fabric Lakehouse (add columns, create tables, change types), the SQL analytics endpoint does not update automatically. New columns remain invisible to SQL queries, Power BI Direct Lake models, and any downstream tool reading from the endpoint. The manual fix (clicking "Refresh Schema" in the portal) takes 10 to 12 minutes and cannot be automated in pipelines.

## The Solution

Use the official Fabric REST API to trigger on-demand metadata refresh:

```
POST /v1/workspaces/{workspaceId}/sqlEndpoints/{sqlEndpointId}/refreshMetadata
```

This repository provides a complete, tested implementation with:
- Automatic workspace and lakehouse context detection
- Long Running Operation (LRO) polling with progress tracking
- Retry logic for HTTP 429 (rate limit) and 5xx (server errors) with exponential backoff
- Per-table sync status reporting
- A reusable `sync_sql_endpoint()` function for any notebook or pipeline

## Repository Contents

```
.
├── generate_voyagehub_data.py                    # Synthetic dataset generator (60M rows)
├── voyagehub_sql_endpoint_refresh_blog.ipynb      # End-to-end Fabric notebook (blog material)
├── fabric_sql_endpoint_sync.ipynb                 # Standalone sync utility (sempy.fabric)
└── fabric_sql_endpoint_refresh_playbook.ipynb     # Diagnostic playbook (notebookutils)
```

### generate_voyagehub_data.py

Generates a 60 million row synthetic dataset for VoyageHub, a travel super-app covering flights, hotels, and rides. Output is Snappy-compressed Parquet files, each under 95MB (GitHub-safe).

**Features:**
- Two CLI modes: `--mode smoke` (1.5M rows, quick test) and `--mode prod` (60M rows)
- Adaptive chunk sizing: calibrates on a pilot chunk to stay under 95MB per file
- 30 columns with realistic distributions across user demographics, booking details, financials, and engagement metrics
- 8 automated data quality assertion rules with JSON audit report
- Privacy mode: `--hash-emails` replaces emails with SHA-256 hashes
- Observability: run summary with throughput metrics, timing, and storage footprint

```bash
# Quick test (1.5M rows, ~25 seconds)
python generate_voyagehub_data.py --mode smoke

# Full dataset (60M rows, ~14 minutes)
python generate_voyagehub_data.py --mode prod

# Privacy mode
python generate_voyagehub_data.py --mode prod --hash-emails
```

**Dependencies:** Python 3.9+, pandas, numpy, pyarrow (all standard)

### voyagehub_sql_endpoint_refresh_blog.ipynb

A Fabric notebook that tells the complete story in 8 phases:

1. **Data Ingestion**: Reads Parquet chunks, writes a partitioned Delta table (by year/month)
2. **EDA**: Null analysis, revenue breakdown, booking trends, membership tier analysis
3. **Feature Engineering**: Adds 10 calculated columns (spend_tier, trip_category, net_revenue, etc.)
4. **The Problem**: Proves the Delta table has 42 columns but the SQL endpoint shows 32
5. **The Solution**: Triggers programmatic schema refresh via `sempy.fabric.FabricRestClient`
6. **Verification**: Confirms all 42 columns are queryable through the SQL endpoint
7. **Reusable Utility**: Clean `sync_sql_endpoint()` function for production use
8. **Troubleshooting**: Common errors, causes, and fixes

### Reference Implementations

| Notebook | Auth Method | Use Case |
|---|---|---|
| `fabric_sql_endpoint_sync.ipynb` | `sempy.fabric.FabricRestClient` | Clean, production-ready sync utility |
| `fabric_sql_endpoint_refresh_playbook.ipynb` | `notebookutils.credentials.getToken` | Diagnostic playbook with detailed context detection |

## Dataset

The full 60M-row VoyageHub dataset (5.1GB, 69 Parquet files) is available on Kaggle:

**[VoyageHub Travel 60M Transactions on Kaggle](https://www.kaggle.com/datasets/sulaimanahmed/voyagehub-travel-superapp-60m-transactions)**

Or generate it locally:

```bash
python generate_voyagehub_data.py --mode prod
```

### Schema (30 columns)

| Column | Type | Description |
|---|---|---|
| `transaction_id` | string | UUID, unique per booking |
| `user_id` | string | USR-{7-digit}, ~2M unique users |
| `user_name` | string | Diverse names (Western, Asian, Middle-Eastern, Latin) |
| `user_email` | string | Derived from name + domain |
| `user_country` | string | 15 countries, weighted (USA 25%, UK 10%, India 10%) |
| `user_membership_tier` | string | Bronze/Silver/Gold/Platinum/Diamond |
| `booking_type` | string | flight (40%), hotel (35%), ride (25%) |
| `booking_date` | date | 2022 to 2025, seasonal peaks |
| `booking_timestamp` | timestamp | UTC, with time-of-day distribution |
| `destination_country` | string | 20 countries (Indonesia 15%, Thailand 12%) |
| `destination_city` | string | Real cities per country |
| `origin_city` | string | Real cities from user's country |
| `booking_status` | string | completed (72%), cancelled, refunded, pending, no_show, in_progress |
| `payment_method` | string | credit_card, debit_card, digital_wallet, bank_transfer, crypto, BNPL |
| `currency` | string | USD (40%), EUR, GBP, SGD, and others |
| `base_amount` | float | Varies by type: flights $80 to $2,500, hotels $25 to $800, rides $3 to $150 |
| `tax_amount` | float | 5% to 18%, varies by destination |
| `discount_amount` | float | 0 for 60%, else $5 to $200 |
| `total_amount` | float | base + tax - discount (deterministic) |
| `promo_code` | string | 20 codes, NULL for 70% |
| `platform` | string | mobile_app, web, partner_api, call_center |
| `device_os` | string | iOS, Android, Windows, macOS, Linux |
| `app_version` | string | 15 versions, newer weighted higher |
| `session_duration_seconds` | int | 30 to 3,600, median varies by booking type |
| `is_repeat_booking` | bool | True for ~35% |
| `rating` | float | 1.0 to 5.0, left-skewed (mean 4.1), NULL for 40% |
| `review_text` | string | 50 phrases, NULL for 80% |
| `cancellation_reason` | string | 6 reasons, NULL unless cancelled/refunded |
| `vendor_id` | string | VND-{5-digit}, ~5,000 vendors |
| `vendor_name` | string | Realistic names by type |

## API Reference

| Endpoint | Documentation |
|---|---|
| Refresh SQL Endpoint Metadata | [learn.microsoft.com](https://learn.microsoft.com/en-us/rest/api/fabric/sqlendpoint/items/refresh-sql-endpoint-metadata) |
| Get Lakehouse | [learn.microsoft.com](https://learn.microsoft.com/en-us/rest/api/fabric/lakehouse/items/get-lakehouse) |
| Long Running Operations | [learn.microsoft.com](https://learn.microsoft.com/en-us/rest/api/fabric/articles/long-running-operation) |
| FabricRestClient | [learn.microsoft.com](https://learn.microsoft.com/en-us/python/api/semantic-link-sempy/sempy.fabric.fabricrestclient) |

## Quick Start

```python
# In any Fabric notebook, after writing to a Delta table:
import sempy.fabric as fabric

client = fabric.FabricRestClient()
ws_id = fabric.get_workspace_id()
lh_id = fabric.get_lakehouse_id()

# Get SQL endpoint ID
lh = client.get(f"/v1/workspaces/{ws_id}/lakehouses/{lh_id}").json()
sql_id = lh["properties"]["sqlEndpointProperties"]["id"]

# Trigger refresh
resp = client.post(
    f"/v1/workspaces/{ws_id}/sqlEndpoints/{sql_id}/refreshMetadata",
    json={"recreateTables": False, "timeout": {"timeUnit": "Minutes", "value": 15}}
)
# Handle 200 (sync) or 202 (async LRO) response
```

## License

MIT
