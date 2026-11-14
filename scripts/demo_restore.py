"""Demo 4 - restore the online table to a pre-incident Delta version.

Scenario:
  1. The on-duty operator accidentally deletes a water-level observation file.
  2. The service restores the table to the version just before the incident:
     target rows (duplicates included) and schema are rewritten into a brand
     new parquet path and swapped in with one dataChange=true Delta commit.
  3. A second restore request carrying the stale (pre-restore) current version
     is rejected with HTTP 409 - a concurrent commit is never overwritten.
  4. Time travel shows every old snapshot, including the incident version,
     remains readable and the transaction log stays contiguous.

Usage:
    python scripts/demo_restore.py
    API_1=http://127.0.0.1:8001 python scripts/demo_restore.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._demo_common import API_1, banner, api, show_snapshot  # noqa: E402


def main() -> None:
    client = api(API_1)
    banner("Demo 4: restore the water-level table after an accidental delete")

    # The restore target is the latest version just before the incident.
    pre_incident = show_snapshot(client)
    target_version = pre_incident["version"]
    victim = pre_incident["active_files"][0]
    print(f"\ntarget restore version: v{target_version}, rows={pre_incident['row_count']}")

    # --- The incident: an external writer deletes a live observation file. ---
    from app.external_writer import delete_files
    from scripts._demo_common import TABLE_DIR

    info = delete_files(TABLE_DIR, [victim])
    print(
        f"\n[incident] delete committed at v{info['version']}: {victim}\n"
    )
    incident = show_snapshot(client)
    assert victim not in incident["active_files"]
    assert incident["row_count"] < pre_incident["row_count"]

    # --- Restore: operator confirms the current version and pins the target. -
    current_version = client.get("/table/versions").json()["latest_version"]
    print(
        f"POST /restore target=v{target_version} "
        f"confirmed_current=v{current_version}"
    )
    r = client.post(
        "/restore",
        json={
            "target_version": target_version,
            "current_version": current_version,
        },
    )
    print(f"-> HTTP {r.status_code}")
    r.raise_for_status()
    body = r.json()
    for key, value in body.items():
        print(f"  {key}: {value}")

    banner("Restored table matches the target version")
    restored = show_snapshot(client)
    assert restored["row_count"] == pre_incident["row_count"]
    assert restored["active_file_count"] == 1  # one brand new parquet path
    assert body["output_file"] in restored["active_files"]
    assert body["rows"] == pre_incident["row_count"]

    banner("Old snapshots stay readable; the log stays contiguous")
    versions = client.get("/table/versions").json()["versions"]
    print(f"contiguous log versions: {versions}")
    assert versions == list(range(0, versions[-1] + 1))
    # The incident snapshot and the pre-incident target are both time-travelable.
    incident_again = show_snapshot(client, version=info["version"])
    assert victim not in incident_again["active_files"]
    target_again = show_snapshot(client, version=target_version)
    assert target_again["row_count"] == pre_incident["row_count"]

    banner("Stale confirmation is rejected with 409 (no overwrite)")
    stale = client.post(
        "/restore",
        json={
            "target_version": target_version,
            "current_version": current_version,  # now behind the latest
        },
    )
    print(f"retry with stale current version -> HTTP {stale.status_code}: {stale.json()}")
    assert stale.status_code == 409
    # The refused restore changed nothing.
    assert client.get("/table/versions").json()["latest_version"] == body["committed_version"]

    print(
        "\nOK: rows, duplicates and schema restored via a dataChange=true "
        "commit on a fresh parquet path; old snapshots remain readable and "
        "concurrent commits are never overwritten."
    )


if __name__ == "__main__":
    main()
