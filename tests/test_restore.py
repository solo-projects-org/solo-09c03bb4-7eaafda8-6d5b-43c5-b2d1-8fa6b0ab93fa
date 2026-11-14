"""End-to-end tests for point-in-time restore of the water-level Delta table."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import delta
from app.external_writer import append_rows, delete_files
from app.sample_data import create_sample_table


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def table(tmp_path: Path) -> Path:
    t = tmp_path / "water_level"
    create_sample_table(t)
    return t


def _rows(t: Path, version: int | None = None) -> list[tuple]:
    return sorted(tuple(r.values()) for r in delta.read_table(t, version=version).to_pylist())


@pytest.fixture()
def client(tmp_path: Path, monkeypatch) -> TestClient:
    data_root = tmp_path / "data"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setenv("TABLE_NAME", "water_level")
    import app.main as main

    monkeypatch.setattr(main, "DATA_ROOT", data_root)
    monkeypatch.setattr(main, "TABLE_DIR", data_root / "water_level")
    with TestClient(main.app) as c:
        yield c


# ---------------------------------------------------------------------------
# Correctness: rows, duplicates, schema and target equivalence
# ---------------------------------------------------------------------------


def test_restore_reproduces_target_rows_and_schema(table: Path) -> None:
    # The on-duty operator accidentally deletes one observation file.
    latest = delta.list_versions(table)[-1]
    victim = sorted(delta.load_snapshot(table).active)[0]
    target_rows = _rows(table, version=2)

    delete_files(table, [victim])
    assert _rows(table) != target_rows  # the incident is visible

    result = delta.restore_table(
        table, target_version=2, expected_current_version=latest + 1
    )

    snap = delta.load_snapshot(table)
    assert snap.version == result["version"] == latest + 2
    restored = delta.read_table(table)
    # Rows, including every duplicate, match the target version exactly.
    assert restored.num_rows == delta.read_table(table, version=2).num_rows
    assert _rows(table) == target_rows
    # Schema is identical to the target version's schema.
    assert restored.schema == delta.read_table(table, version=2).schema
    assert restored.schema == delta.TABLE_SCHEMA
    # The target active file set is represented by exactly one new file.
    assert len(snap.active) == 1
    assert result["output_file"] in snap.active
    assert result["rows"] == restored.num_rows


def test_restore_preserves_explicit_duplicate_rows(table: Path) -> None:
    dup_rows = [
        {"station": "ST-DUP", "utc_seconds": 100, "level_mm": 5},
        {"station": "ST-DUP", "utc_seconds": 100, "level_mm": 5},
    ]
    append_rows(table, dup_rows)
    append_rows(table, list(dup_rows))
    dup_version = delta.list_versions(table)[-1]
    assert _rows(table, version=dup_version).count(("ST-DUP", 100, 5)) == 4
    # More appends, then an accidental delete: restore target is the version
    # that already contains both duplicated files.
    append_rows(
        table, [{"station": "ST-009", "utc_seconds": 7, "level_mm": 9}]
    )
    victim = sorted(delta.load_snapshot(table).active)[-1]
    delete_files(table, [victim])

    target_rows = _rows(table, version=dup_version)
    result = delta.restore_table(
        table,
        target_version=dup_version,
        expected_current_version=delta.list_versions(table)[-1],
    )

    after = _rows(table)
    assert after == target_rows
    assert after.count(("ST-DUP", 100, 5)) == 4  # no dedup happened
    assert result["rows"] == len(after)


def test_restore_uses_brand_new_paths_and_keeps_old_snapshot_files(table: Path) -> None:
    target_version = 2
    target_files = sorted(delta.load_snapshot(table, target_version).active)
    before_files = {p.name for p in table.glob("*.parquet")}
    current = delta.list_versions(table)[-1]

    result = delta.restore_table(
        table, target_version, expected_current_version=current
    )

    # The restore output is a fresh path that never existed before.
    assert result["output_file"] not in target_files
    assert Path(result["output_file"]).name not in before_files
    # All physical files of the target snapshot remain on disk (time travel).
    for rel in target_files:
        assert (table / rel).is_file()
    # Old snapshot is byte-identical in content and active file set.
    old_snap = delta.load_snapshot(table, version=target_version)
    assert sorted(old_snap.active) == target_files
    assert _rows(table, version=target_version) == _rows(table, version=result["version"])


def test_restore_old_snapshots_remain_readable(table: Path) -> None:
    rows_by_version = {v: _rows(table, version=v) for v in delta.list_versions(table)}
    current = delta.list_versions(table)[-1]

    delta.restore_table(table, 1, expected_current_version=current)

    for v, rows in rows_by_version.items():
        assert _rows(table, version=v) == rows


def test_restore_commit_is_single_data_change_true_swap(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    current_active = sorted(delta.load_snapshot(table).active)
    result = delta.restore_table(
        table, target_version=2, expected_current_version=current
    )

    actions = delta.read_log(table, result["version"])
    adds = [a["add"] for a in actions if "add" in a]
    removes = [a["remove"] for a in actions if "remove" in a]
    infos = [a["commitInfo"] for a in actions if "commitInfo" in a]
    assert len(adds) == 1 and adds[0]["dataChange"] is True
    assert len(removes) == len(current_active)
    assert all(r["dataChange"] is True for r in removes)
    assert sorted(r["path"] for r in removes) == current_active
    assert infos[0]["operation"] == "RESTORE"
    # The log stays contiguous and gains exactly one new version.
    versions = delta.list_versions(table)
    assert versions == list(range(0, versions[-1] + 1))
    assert versions[-1] == current + 1


def test_restore_to_empty_snapshot_is_valid(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    active = sorted(delta.load_snapshot(table).active)
    delete_files(table, active)
    empty_version = delta.list_versions(table)[-1]
    assert delta.load_snapshot(table, empty_version).active == {}
    # Re-populate, then restore back to the empty snapshot.
    append_rows(
        table, [{"station": "ST-X", "utc_seconds": 1, "level_mm": 2}]
    )
    latest = delta.list_versions(table)[-1]
    assert delta.load_snapshot(table, latest).active  # data is back

    result = delta.restore_table(
        table, empty_version, expected_current_version=latest
    )
    snap = delta.load_snapshot(table)
    # The restored snapshot holds one brand new empty parquet file and no rows.
    assert len(snap.active) == 1
    assert result["output_file"] in snap.active
    assert delta.read_table(table).num_rows == 0
    assert result["rows"] == 0
    # Empty restore still writes one dataChange=true add and one remove.
    actions = delta.read_log(table, result["version"])
    assert len([a for a in actions if "add" in a]) == 1
    assert len([a for a in actions if "remove" in a]) == 1
    assert current < result["version"]
    # Time travel to the populated version still works.
    assert delta.read_table(table, version=latest).num_rows == 1


# ---------------------------------------------------------------------------
# Concurrency: any intervening commit must reject the restore
# ---------------------------------------------------------------------------


def test_restore_rejected_when_table_advanced_by_append(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    append_rows(
        table, [{"station": "ST-NEW", "utc_seconds": 99, "level_mm": 1}]
    )
    advanced = delta.list_versions(table)[-1]
    before_snap = sorted(delta.load_snapshot(table).active)
    before_rows = _rows(table)

    with pytest.raises(delta.ConflictError):
        delta.restore_table(table, 2, expected_current_version=current)

    # Table is completely untouched; the new observation survives.
    snap = delta.load_snapshot(table)
    assert snap.version == advanced
    assert sorted(snap.active) == before_snap
    assert _rows(table) == before_rows


def test_restore_rejected_when_table_advanced_by_compaction(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    chosen = sorted(delta.load_snapshot(table).active)[:2]
    delta.compact(table, chosen)

    with pytest.raises(delta.ConflictError):
        delta.restore_table(table, 2, expected_current_version=current)


def test_restore_rejected_when_table_advanced_by_delete(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    victim = sorted(delta.load_snapshot(table).active)[0]
    delete_files(table, [victim])

    with pytest.raises(delta.ConflictError):
        delta.restore_table(table, 2, expected_current_version=current)
    assert victim not in delta.load_snapshot(table).active


def _worker_restore(table_str: str, target: int, current: int, q) -> None:  # pragma: no cover
    try:
        q.put(("ok", delta.restore_table(Path(table_str), target, current)["version"]))
    except delta.DeltaError as exc:
        q.put(("conflict", str(exc)))


def test_two_processes_restore_same_base_only_one_commits(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    ctx = multiprocessing.get_context("spawn")
    q1, q2 = ctx.Queue(), ctx.Queue()
    args = (str(table), 2, current)
    p1 = ctx.Process(target=_worker_restore, args=(*args, q1))
    p2 = ctx.Process(target=_worker_restore, args=(*args, q2))
    p1.start(); p2.start()
    p1.join(30); p2.join(30)
    assert p1.exitcode == 0 and p2.exitcode == 0
    r1, r2 = q1.get(), q2.get()
    statuses = {r1[0], r2[0]}
    assert statuses == {"ok", "conflict"}
    # Exactly one new contiguous log version; the loser changed nothing.
    versions = delta.list_versions(table)
    assert versions == list(range(0, versions[-1] + 1))
    assert versions[-1] == current + 1
    # The winner restored the target content.
    assert _rows(table) == _rows(table, version=2)
    # Loser's output file was rolled back: only the committed restore file.
    snap = delta.load_snapshot(table)
    assert len(snap.active) == 1


def test_restore_validates_versions(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    with pytest.raises(delta.VersionNotFoundError):
        delta.restore_table(table, 99, expected_current_version=current)
    with pytest.raises(delta.VersionNotFoundError):
        delta.restore_table(table, 2, expected_current_version=99)
    with pytest.raises(delta.InvalidRequestError):
        delta.restore_table(table, current, expected_current_version=2)


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_missing_target_file_aborts_without_changing_table(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    target_files = sorted(delta.load_snapshot(table, version=2).active)
    victim = table / target_files[0]
    victim.unlink()  # simulate a lost physical parquet needed by the target

    before_snap = delta.load_snapshot(table)
    with pytest.raises(delta.TableCorruptError, match="missing on disk"):
        delta.restore_table(table, 2, expected_current_version=current)

    # The committed table is completely untouched: same version and file set.
    snap = delta.load_snapshot(table)
    assert snap.version == current
    assert sorted(snap.active) == sorted(before_snap.active)
    # No half-finished output leaked into the table directory.
    assert all(not p.name.startswith("restore-") for p in table.rglob("*.parquet"))
    staging = table / delta.STAGING_DIR
    assert not staging.exists() or not any(staging.iterdir())


def test_fault_after_data_write_leaves_no_snapshot_change(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    before_snap = delta.load_snapshot(table)
    before_rows = _rows(table)

    with pytest.raises(RuntimeError):
        delta.restore_table(
            table, 2, current, fault_inject="after_data_write"
        )

    snap = delta.load_snapshot(table)
    assert snap.version == current
    assert sorted(snap.active) == sorted(before_snap.active)
    assert _rows(table) == before_rows
    # The half-written output is neither queryable nor left at a data path.
    assert all(rel in before_snap.active for rel in snap.active)
    data_files = {p.name for p in table.glob("*.parquet")}
    assert data_files == {Path(rel).name for rel in before_snap.active}
    staging = list((table / delta.STAGING_DIR).glob("*")) if (table / delta.STAGING_DIR).exists() else []
    assert all(p.name.startswith("restore-") for p in staging)


def test_successful_restore_leaves_no_staging_orphans(table: Path) -> None:
    current = delta.list_versions(table)[-1]
    delta.restore_table(table, 2, expected_current_version=current)
    staging = table / delta.STAGING_DIR
    assert not staging.exists() or not any(staging.iterdir())


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def test_api_restore_endpoint(client: TestClient) -> None:
    versions = client.get("/table/versions").json()["versions"]
    current = versions[-1]
    target_rows = client.get(
        "/table/snapshot", params={"version": 2}
    ).json()["row_count"]

    r = client.post(
        "/restore",
        json={"target_version": 2, "current_version": current},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["committed_version"] == current + 1
    assert body["target_version"] == 2
    assert body["confirmed_current_version"] == current
    assert body["files_added"] == 1
    assert body["rows"] == target_rows

    latest = client.get("/table/snapshot").json()
    assert latest["row_count"] == target_rows
    assert latest["active_file_count"] == 1
    # Old snapshots remain readable through the API.
    old = client.get("/table/snapshot", params={"version": current}).json()
    assert old["active_file_count"] == 4


def test_api_restore_conflict_returns_409(client: TestClient) -> None:
    from app import main

    current = client.get("/table/versions").json()["latest_version"]
    # An external writer commits between the operator's confirmation and request.
    append_rows(
        main.TABLE_DIR,
        [{"station": "ST-NEW", "utc_seconds": 99, "level_mm": 1}],
    )
    r = client.post(
        "/restore",
        json={"target_version": 2, "current_version": current},
    )
    assert r.status_code == 409
    assert "refused" in r.json()["detail"]


def test_api_restore_validation_and_404_mapping(client: TestClient) -> None:
    current = client.get("/table/versions").json()["latest_version"]
    # 422: pydantic rejects negative versions.
    r = client.post(
        "/restore", json={"target_version": -1, "current_version": current}
    )
    assert r.status_code == 422
    # 404: unknown target version (confirmed current version is still valid).
    r = client.post(
        "/restore", json={"target_version": 77, "current_version": current}
    )
    assert r.status_code == 404
    # 400: target newer than the confirmed current version.
    r = client.post(
        "/restore", json={"target_version": current, "current_version": 1}
    )
    assert r.status_code == 400
    # 404: unknown confirmed current version.
    r = client.post(
        "/restore", json={"target_version": 1, "current_version": 77}
    )
    assert r.status_code == 404


def test_api_restore_after_accidental_delete_scenario(client: TestClient) -> None:
    """Full incident flow: delete -> restore -> forecasting reads target data."""
    from app import main

    target = client.get("/table/snapshot", params={"version": 2}).json()
    # The operator accidentally deletes two files that were active at v2.
    victims = target["active_files"][:2]
    live_before = client.get("/table/snapshot").json()
    assert live_before["row_count"] >= target["row_count"]
    delete_files(main.TABLE_DIR, victims)
    after_delete = client.get("/table/snapshot").json()
    assert after_delete["row_count"] < live_before["row_count"]

    current = client.get("/table/versions").json()["latest_version"]
    r = client.post(
        "/restore",
        json={"target_version": target["version"], "current_version": current},
    )
    assert r.status_code == 200, r.text
    restored = client.get("/table/snapshot").json()
    assert restored["row_count"] == target["row_count"]
    assert restored["version"] == r.json()["committed_version"]
    # Swagger documents the new route.
    paths = client.get("/openapi.json").json()["paths"]
    assert "/restore" in paths
