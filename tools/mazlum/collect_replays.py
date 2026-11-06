"""Mazlum — mine Kaggle replays where the target deck won.

Target deck comes from ``decks/meta/README.md``: **Banette** is the highest
win-rate archetype (63.2%, 36-21 over 188 mined episodes), so Mazlum is trained
on Banette wins only.

How it works (avoids downloading every replay):

1. Leaderboard -> teams -> public submissions.
2. Download **one** replay per submission to learn that submission's deck.
3. If the deck archetype matches ``--archetype``, the episode listing already
   tells us which side won (``agents[].reward``), so only winning episodes are
   downloaded and kept in ``replays/``.

Progress is recorded in ``replays/_collected.json`` so reruns resume.

Usage::

    python tools/mazlum/collect_replays.py
    python tools/mazlum/collect_replays.py --target 100 --archetype Banette
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import replay_meta as rm  # noqa: E402

COMPETITION = "the-pokemon-company-ptcg-ai-battle-challenge-playground"
DEFAULT_OUT = ROOT / "replays"
MANIFEST_NAME = "_collected.json"
TMP_DIR = ROOT / "replays" / "_tmp"

_local = threading.local()
_lock = threading.Lock()

# Seconds to sleep before each replay request (polite single-worker mode).
_DELAY = 0.0


def api_client():
    """One authenticated Kaggle client per thread."""
    client = getattr(_local, "api", None)
    if client is None:
        from kaggle.api.kaggle_api_extended import KaggleApi

        client = KaggleApi()
        client.authenticate()
        _local.api = client
    return client


def load_manifest(path: Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"target_archetype": None, "episodes": [], "submissions": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def list_teams(limit: int) -> list[tuple[int, str]]:
    """Read the full leaderboard CSV (hundreds of teams) instead of one page."""
    import csv
    import io
    import tempfile
    import zipfile

    client = api_client()
    with tempfile.TemporaryDirectory() as tmp:
        client.competition_leaderboard_download(COMPETITION, path=tmp)
        archive = next(Path(tmp).glob("*.zip"))
        with zipfile.ZipFile(archive) as zf:
            name = zf.namelist()[0]
            rows = list(csv.DictReader(io.StringIO(zf.read(name).decode("utf-8-sig"))))

    teams: list[tuple[int, str]] = []
    for row in rows:
        team_id = row.get("TeamId") or row.get("teamId")
        team_name = row.get("TeamName") or row.get("teamName") or "?"
        if team_id:
            teams.append((int(team_id), str(team_name)))
        if len(teams) >= limit:
            break
    return teams


def _api_retry(call, *args, attempts: int = 8, base: float = 8.0, **kwargs):
    """Retry a Kaggle API call with backoff on 429 (team/submission listings)."""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return call(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            text = str(exc)
            if "429" not in text and "Too Many Requests" not in text:
                raise
            pause = base * (attempt + 1) + random.random() * 2.0
            print(f"    429 on api call; sleeping {pause:.1f}s (attempt {attempt + 1}/{attempts})", flush=True)
            time.sleep(pause)
    if last is not None:
        raise last
    return None


def list_submissions(team_id: int, limit: int) -> list[int]:
    subs = _api_retry(api_client().competition_team_submissions, team_id) or []
    return [int(sub.id) for sub in subs[:limit]]


def list_episodes(submission_id: int) -> list[dict]:
    episodes = _api_retry(api_client().competition_list_episodes, submission_id) or []
    rows = []
    for episode in episodes:
        if "COMPLETED" not in str(getattr(episode, "state", "")):
            continue
        agents = []
        for agent in getattr(episode, "agents", None) or []:
            agents.append(
                {
                    "index": int(getattr(agent, "index", -1)),
                    "reward": float(getattr(agent, "reward", 0) or 0),
                    "team": str(getattr(agent, "teamName", "?")),
                    "submission_id": int(getattr(agent, "submissionId", -1) or -1),
                }
            )
        rows.append({"id": int(episode.id), "agents": agents})
    return rows


def download_replay(episode_id: int, out_dir: Path, attempts: int = 6) -> Path:
    """Download one replay, backing off exponentially when Kaggle answers 429."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"episode-{episode_id}-replay.json"
    if not target.is_file():
        if _DELAY > 0:
            time.sleep(_DELAY)
        delay = 8.0
        for attempt in range(attempts):
            try:
                api_client().competition_episode_replay(episode_id, path=str(out_dir), quiet=True)
                break
            except Exception as exc:  # noqa: BLE001
                rate_limited = "429" in str(exc) or "Too Many Requests" in str(exc)
                if not rate_limited or attempt == attempts - 1:
                    raise
                pause = delay + random.uniform(0, 3)
                print(f"    429 on {episode_id}; sleeping {pause:.1f}s (attempt {attempt + 1}/{attempts})")
                time.sleep(pause)
                delay = min(delay * 2, 180.0)
    if not target.is_file():
        raise FileNotFoundError(f"replay not downloaded: {episode_id}")
    return target


def deck_archetypes(path: Path) -> tuple[list[str], int | None, list[str]]:
    """Return (archetypes per player, winner index, team names)."""
    episode = json.loads(path.read_text(encoding="utf-8"))
    decks = rm.extract_decks(episode)
    if not decks:
        return [], None, rm.team_names(episode)[:2]
    return [rm.archetype(deck) for deck in decks], rm.winner_index(episode), rm.team_names(episode)[:2]


def main() -> int:
    # Windows consoles default to cp1252; team names are Unicode.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="Collect replays where a target deck won")
    parser.add_argument("--competition", default=COMPETITION)
    parser.add_argument("--archetype", default=None, help="single target deck archetype")
    parser.add_argument(
        "--archetypes",
        default=None,
        help="comma-separated archetypes for data diversity (e.g. Banette,Alakazam,Dragapult)",
    )
    parser.add_argument("--target", type=int, default=100, help="number of winning replays to keep")
    parser.add_argument("--teams", type=int, default=40, help="leaderboard teams to scan")
    parser.add_argument("--subs-per-team", type=int, default=2, help="submissions per team")
    parser.add_argument("--workers", type=int, default=4, help="parallel replay downloads")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between replay requests")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    global _DELAY
    _DELAY = max(0.0, args.delay)

    targets: set[str] = set()
    if args.archetypes:
        targets |= {name.strip() for name in args.archetypes.split(",") if name.strip()}
    if args.archetype:
        targets.add(args.archetype)
    if not targets:
        targets = {"Banette"}

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    manifest.setdefault("submissions", {})
    manifest["target_archetypes"] = sorted(targets)
    manifest["target_archetype"] = next(iter(targets)) if len(targets) == 1 else None

    kept = {int(entry["episode_id"]) for entry in manifest["episodes"]}
    print(f"targets: {', '.join(sorted(targets))}  keep: {args.target}  existing: {len(kept)}")

    teams = list_teams(args.teams)
    print(f"teams scanned from leaderboard: {len(teams)}")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    scanned = 0
    probe_failures = 0

    for team_id, team_name in teams:
        if len(kept) >= args.target:
            break
        try:
            submissions = list_submissions(team_id, args.subs_per_team)
        except Exception as exc:  # noqa: BLE001
            print(f"  {team_name}: submissions failed ({exc})")
            continue

        for submission_id in submissions:
            if len(kept) >= args.target:
                break
            try:
                episodes = list_episodes(submission_id)
            except Exception as exc:  # noqa: BLE001
                print(f"  sub {submission_id}: episodes failed ({exc})")
                continue
            if not episodes:
                continue

            scanned += 1
            # One probe replay tells us this submission's deck; cache it so a
            # resumed run does not download the same probe again.
            cached = manifest["submissions"].get(str(submission_id))
            if cached:
                archetypes = list(cached)
            else:
                probe_dir = TMP_DIR / f"probe-{submission_id}"
                try:
                    probe_path = download_replay(episodes[0]["id"], probe_dir)
                    archetypes, _, _ = deck_archetypes(probe_path)
                except Exception as exc:  # noqa: BLE001
                    probe_failures += 1
                    print(f"  sub {submission_id}: probe failed ({exc})")
                    shutil.rmtree(probe_dir, ignore_errors=True)
                    continue
                finally:
                    shutil.rmtree(probe_dir, ignore_errors=True)
                manifest["submissions"][str(submission_id)] = archetypes
                save_manifest(manifest_path, manifest)

            if not (set(archetypes) & targets):
                continue

            # A submission's deck is fixed, but matchmaking puts it on side 0 or
            # 1 per episode, so resolve the side from the episode metadata.
            def side_for(row: dict) -> int | None:
                for agent in row["agents"]:
                    if agent.get("submission_id") == submission_id:
                        return int(agent["index"])
                for agent in row["agents"]:
                    if agent.get("team") == team_name:
                        return int(agent["index"])
                return None

            probe_agents = episodes[0]["agents"]
            has_identity = any(
                agent.get("submission_id", -1) >= 0 or agent.get("team") not in ("?", "", None)
                for agent in probe_agents
            )

            if has_identity:
                probe_side = side_for(episodes[0])
                if probe_side is None or probe_side >= len(archetypes) or archetypes[probe_side] not in targets:
                    continue
                wins = [
                    row
                    for row in episodes
                    if len(row["agents"]) == 2
                    and (row_side := side_for(row)) is not None
                    and row_side < len(row["agents"])
                    and row["agents"][row_side]["reward"] > 0
                ]
                label = f"{archetypes[probe_side]} side {probe_side}"
            else:
                # Older episodes strip submission/team metadata. Download and
                # verify the winner's archetype from the replay itself.
                if not (set(archetypes) & targets):
                    continue
                wins = list(episodes)
                label = f"{archetypes} (metadata-less, verify on download)"

            print(
                f"  {team_name} sub {submission_id}: {label}, {len(wins)} candidate episodes"
            )

            todo = [row["id"] for row in wins if row["id"] not in kept][: max(0, args.target - len(kept))]

            def harvest(episode_id: int):
                path = download_replay(episode_id, TMP_DIR / f"ep-{episode_id}")
                return episode_id, path

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(harvest, episode_id): episode_id for episode_id in todo}
                for future in as_completed(futures):
                    episode_id = futures[future]
                    try:
                        _, path = future.result()
                    except Exception as exc:  # noqa: BLE001
                        print(f"    episode {episode_id}: download failed ({exc})")
                        continue

                    try:
                        archetypes, winner, names = deck_archetypes(path)
                    except Exception as exc:  # noqa: BLE001
                        print(f"    episode {episode_id}: parse failed ({exc})")
                        continue

                    if winner is None or winner >= len(archetypes) or archetypes[winner] not in targets:
                        continue

                    final = args.out / path.name
                    shutil.move(str(path), final)
                    shutil.rmtree(path.parent, ignore_errors=True)
                    with _lock:
                        if episode_id in kept:
                            continue
                        kept.add(episode_id)
                        manifest["episodes"].append(
                            {
                                "episode_id": episode_id,
                                "submission_id": submission_id,
                                "team": team_name,
                                "winner_side": winner,
                                "archetypes": archetypes,
                                "teams": names,
                                "file": final.name,
                            }
                        )
                        save_manifest(manifest_path, manifest)
                    count = len(kept)
                    print(f"    kept {count}/{args.target}: episode {episode_id} ({archetypes[0]} vs {archetypes[1]})")
                    if count >= args.target:
                        break

            if len(kept) >= args.target:
                break

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    save_manifest(manifest_path, manifest)

    print(f"\nsubmissions scanned: {scanned}  probe failures: {probe_failures}")
    print(f"kept: {len(kept)} wins ({len(targets)} archetypes) -> {args.out}")
    print(f"manifest: {manifest_path}")
    if len(kept) < args.target:
        print(f"warning: only {len(kept)} of {args.target} requested replays were found")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
