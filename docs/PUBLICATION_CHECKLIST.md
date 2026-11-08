# Before any GitHub upload

Preparation performs no network write. Only this `repository/` tree is the intended upload root; its parent contains private records that must not be committed.

- [ ] Owner/organization and repository name confirmed.
- [ ] Public/private visibility and whether unpublished manuscript information may be disclosed confirmed.
- [ ] Software and trained-weight license selected by the rights holder; applicable third-party terms reviewed. `LICENSE_PENDING.md` is not a license grant.
- [ ] Scope of per-protein IDs, score tables and model weights for disclosure confirmed.
- [ ] Manuscript title, author order, citation and contact details supplied; no DOI invented.
- [ ] All six weight exports and18 primary results retained; limitations and negative results retained.
- [ ] `python scripts/verify_release.py` and CPU tests pass on the exact intended contents.
- [ ] No private audit material, server endpoint, account secret or raw large dataset staged.
- [ ] Authorized GitHub authentication established; no login password pasted into code or repository files.
- [ ] Commit/tag recorded only after upload authorization; proposed tag is `v2.2`, not created by preparation.

Normal Git repositories warn above50MiB/file and block above100MiB/file; this package's verification is stricter and rejects any payload file at50MiB or larger. If scope changes, review Git LFS or release assets rather than silently adding giant files. [GitHub large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Git-over-HTTPS does not use the account login password. Prefer browser/device authorization or a scoped token via a credential manager, or an approved SSH key; do not save tokens in remote URLs, source or logs. [GitHub authentication documentation](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github).

After choosing licensing/citation/repository metadata, regenerate the repository checksums and review the resulting change before uploading. Do not call a content-modified tree the exact previously frozen artifact without a new package digest. Scientific model/version identity need not change for documentation-only corrections, but packaging revisions must be recorded.
