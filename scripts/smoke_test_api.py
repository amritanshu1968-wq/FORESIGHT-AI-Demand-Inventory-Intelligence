"""
PROJECT FORESIGHT — API Live Smoke Test Script
Executes real HTTP requests against the live Uvicorn service and prints status codes and payloads.
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"


def send_get(endpoint: str):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(url, headers={"User-Agent": "SmokeTest"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"\n[GET {endpoint}] -> HTTP {resp.status}")
            print(json.dumps(data, indent=2))
            return resp.status, data
    except urllib.error.HTTPError as e:
        print(f"\n[GET {endpoint}] -> HTTP {e.code}")
        print(e.read().decode("utf-8"))
        return e.code, None


def send_post(endpoint: str, payload: dict):
    url = f"{BASE_URL}{endpoint}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "SmokeTest"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"\n[POST {endpoint}] -> HTTP {resp.status}")
            print(json.dumps(data, indent=2))
            return resp.status, data
    except urllib.error.HTTPError as e:
        print(f"\n[POST {endpoint}] -> HTTP {e.code}")
        print(e.read().decode("utf-8"))
        return e.code, None


if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING LIVE API SMOKE TEST AGAINST UVICORN (127.0.0.1:8000)")
    print("=" * 60)

    # 1. Root & Health
    send_get("/")
    send_get("/health")

    # 2. Real SKU Forecast
    send_get("/forecast/SKU-0272")

    # 3. Real SKU Risk Score
    send_get("/risk-score/SKU-0272")

    # 4. Real Combined SKU View
    send_get("/sku/SKU-0272")

    # 5. Batch Score
    send_post("/batch-score", {"sku_ids": ["SKU-0272", "SKU-0280", "SKU-0036"]})

    # 6. Unknown SKU 404 test
    send_get("/forecast/SKU-9999")

    print("\n" + "=" * 60)
    print("API SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
