#!/bin/bash
# Verify that a run starts from a committed version of the code: a clean tree whose HEAD is a commit of the main branch.
#
# Usage: check_git_clean.sh [repo_root]  (repo_root defaults to $(pwd))
#
# The main branch is the one of TCIRL_MAIN_SOURCE when it is set (another repository, as a path or a URL), else repo_root's own main.
# A HEAD behind main passes with a warning, so the later stages of a long run pass after main moves on; a HEAD with commits that main lacks is refused, and so is a TCIRL_MAIN_SOURCE that cannot be reached.
# Environment overrides:
#   ALLOW_DIRTY=1            Skip every check with a warning, for a trial run; reported numbers come from runs that pass the checks.
#   TCIRL_MAIN_SOURCE=<repo> Another repository whose main branch must contain HEAD.
set -euo pipefail

REPO_ROOT="${1:-$(pwd)}"

if [[ "${ALLOW_DIRTY:-}" == "1" ]]; then
    echo "WARNING: ALLOW_DIRTY=1, skipping the git checks for $REPO_ROOT; this is a trial run."
    exit 0
fi

STATUS=$(git -C "$REPO_ROOT" status --porcelain)
if [[ -n "$STATUS" ]]; then
    echo "Error: git working tree is not clean in $REPO_ROOT. Commit or stash changes before running the reproduction scripts." >&2
    echo "Set ALLOW_DIRTY=1 to override." >&2
    exit 1
fi

if [[ -n "${TCIRL_MAIN_SOURCE:-}" ]]; then
    MAIN_NAME="main of $TCIRL_MAIN_SOURCE"
    # Batch mode and a timeout, so that a host-key or password prompt cannot hang the launcher.
    export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10"
    if ! timeout 60 git -C "$REPO_ROOT" fetch --quiet "$TCIRL_MAIN_SOURCE" main; then
        echo "Error: cannot fetch the main branch of TCIRL_MAIN_SOURCE=$TCIRL_MAIN_SOURCE into $REPO_ROOT, so the run cannot be tied to a commit of main." >&2
        echo "Fix the source, unset TCIRL_MAIN_SOURCE to check against $REPO_ROOT's own main, or set ALLOW_DIRTY=1 for a trial run." >&2
        exit 1
    fi
    MAIN_SHA=$(git -C "$REPO_ROOT" rev-parse FETCH_HEAD)
else
    MAIN_NAME="the main branch of $REPO_ROOT"
    if ! MAIN_SHA=$(git -C "$REPO_ROOT" rev-parse --verify --quiet refs/heads/main); then
        echo "Error: $REPO_ROOT has no main branch to check HEAD against; commit the tree on a branch named main, or set ALLOW_DIRTY=1 for a trial run." >&2
        exit 1
    fi
fi
HEAD_SHA=$(git -C "$REPO_ROOT" rev-parse HEAD)
if [[ "$HEAD_SHA" == "$MAIN_SHA" ]]; then
    exit 0
fi
if git -C "$REPO_ROOT" merge-base --is-ancestor "$HEAD_SHA" "$MAIN_SHA"; then
    BEHIND=$(git -C "$REPO_ROOT" rev-list --count "$HEAD_SHA..$MAIN_SHA")
    echo "WARNING: $REPO_ROOT is at ${HEAD_SHA:0:7}, $BEHIND commit(s) behind $MAIN_NAME (${MAIN_SHA:0:7}); the run uses that older commit of main."
    exit 0
fi
echo "Error: $REPO_ROOT is at ${HEAD_SHA:0:7}, which is not a commit of $MAIN_NAME (${MAIN_SHA:0:7})." >&2
echo "A run comes from main: commit the change to main and bring it here, or set ALLOW_DIRTY=1 for a trial run." >&2
exit 1
