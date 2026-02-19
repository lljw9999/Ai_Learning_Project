#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/apply_github_hardening.sh [owner/repo]
# If omitted, repo slug is inferred from git remote origin.

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required. Install GitHub CLI first." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI is not authenticated. Run: gh auth login" >&2
  exit 1
fi

repo_slug="${1:-}"
if [[ -z "${repo_slug}" ]]; then
  remote_url="$(git remote get-url origin)"
  # Supports https://github.com/owner/repo.git and git@github.com:owner/repo.git
  repo_slug="$(echo "${remote_url}" | sed -E 's#(https://github.com/|git@github.com:)([^/]+/[^/.]+)(\.git)?#\2#')"
fi

if [[ ! "${repo_slug}" =~ .+/.+ ]]; then
  echo "Unable to resolve repository slug. Provide explicitly, e.g. owner/repo." >&2
  exit 1
fi

owner="${repo_slug%%/*}"
repo="${repo_slug##*/}"

echo "Applying branch protection to ${owner}/${repo} (branch: main)..."

# Require PRs + status checks.
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/${owner}/${repo}/branches/main/protection" \
  --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "checks": [
      {"context": "test"},
      {"context": "build"}
    ]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "required_linear_history": false,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "block_creations": false,
  "required_conversation_resolution": true,
  "lock_branch": false,
  "allow_fork_syncing": true
}
JSON

echo "Enabling automatic dependency graph and vulnerability alerts..."

# Best-effort enablement; some settings may depend on plan/permissions.
set +e
gh api --method PUT -H "Accept: application/vnd.github+json" "/repos/${owner}/${repo}/vulnerability-alerts" >/dev/null 2>&1
vuln_status=$?
set -e

if [[ ${vuln_status} -ne 0 ]]; then
  echo "Warning: Could not enable vulnerability alerts automatically (permission/plan dependent)."
fi

echo "Repository hardening completed for ${owner}/${repo}."
