# GitHub Repository Hardening Checklist

Use this once in repository settings.

## 1) Branch Protection (`main`)

Recommended rules:

- Require pull request before merging
- Require approvals: 1+
- Dismiss stale approvals when new commits are pushed
- Require status checks to pass before merging:
  - `test` (from CI workflow)
  - `build` (from Docker Image workflow)
- Require linear history (optional)
- Include administrators (optional)

## 2) Security Features

- Enable Dependabot alerts
- Enable Dependabot security updates
- Enable Secret scanning

## 3) Workflow Permissions

- Default `GITHUB_TOKEN` permissions: read repository contents
- Allow workflows to create/approve PRs only if needed

## 4) Repository Settings

- Enable Discussions for support/Q&A
- Enforce issue and PR templates
- Keep default branch as `main`

