#!/usr/bin/env bash
# scripts/release.sh — bump version, run QA gate, commit, tag, push.
#
# Usage:
#   ./scripts/release.sh patch    # 0.2.4 → 0.2.5  (bug fixes, small additions)
#   ./scripts/release.sh minor    # 0.2.4 → 0.3.0  (new features, backwards-compatible)
#   ./scripts/release.sh major    # 0.2.4 → 1.0.0  (breaking changes)
#
# The script will:
#   1. Read current version from Cargo.toml
#   2. Compute next version based on <bump> arg
#   3. Run: cargo test --workspace (fails fast)
#   4. Update Cargo.toml version
#   5. git commit + git tag vX.Y.Z
#   6. git push origin main + push tag
#
# Exit codes:
#   0  — release complete
#   1  — bad args or QA failure (nothing pushed)

set -euo pipefail

# ── Args ──────────────────────────────────────────────────────────────────────

BUMP="${1:-}"
if [[ -z "$BUMP" || ! "$BUMP" =~ ^(patch|minor|major)$ ]]; then
    echo "Usage: $0 patch|minor|major" >&2
    exit 1
fi

# ── Current version ───────────────────────────────────────────────────────────

CURRENT=$(grep '^version = ' Cargo.toml | head -1 | sed 's/version = "//;s/"//')
if [[ -z "$CURRENT" ]]; then
    echo "error: cannot find version in Cargo.toml" >&2
    exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "$BUMP" in
    patch) PATCH=$((PATCH + 1)) ;;
    minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
    major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
esac

NEXT="${MAJOR}.${MINOR}.${PATCH}"
TAG="v${NEXT}"

echo "Releasing: ${CURRENT} → ${NEXT} (${TAG})"
echo ""

# ── Guard: clean working tree ─────────────────────────────────────────────────

if [[ -n "$(git status --porcelain)" ]]; then
    echo "error: working tree is dirty — commit or stash changes first" >&2
    exit 1
fi

# ── Guard: tag must not already exist ────────────────────────────────────────

if git rev-parse "$TAG" &>/dev/null; then
    echo "error: tag $TAG already exists locally — delete it first or bump again" >&2
    exit 1
fi

# ── QA gate ───────────────────────────────────────────────────────────────────

echo "Running: cargo test --workspace"
cargo test --workspace

# ── Bump version ─────────────────────────────────────────────────────────────

sed -i '' "s/^version = \"${CURRENT}\"/version = \"${NEXT}\"/" Cargo.toml
# Cargo.lock updates on next build; include it so CI doesn't get a lock-file mismatch.
cargo check -q --workspace 2>/dev/null || true

# ── Commit, tag, push ────────────────────────────────────────────────────────

git add Cargo.toml Cargo.lock
git commit -m "chore: release ${TAG}"
git tag "${TAG}"
git push origin main
git push origin "${TAG}"

echo ""
echo "✓ Released ${TAG}"
echo "  CI: https://github.com/stuinfla/learner-rv/actions"
echo "  Release: https://github.com/stuinfla/learner-rv/releases/tag/${TAG}"
