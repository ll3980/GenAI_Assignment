# Apply the organized update to GitHub

The prepared branch reorganizes the existing repository and integrates
Assignment 3. Its code and HTTP checks are documented under
`results/integration/`. Publishing requires authenticated write access to
`ll3980/GenAI_Assignment`.

## Apply the provided patch

Download `GenAI_Assignment_Update.patch` into a directory on your computer.
Open a terminal in that directory, then:

```bash
git clone https://github.com/ll3980/GenAI_Assignment.git
cd GenAI_Assignment
git switch -c assignment3-organized
git apply --check ../GenAI_Assignment_Update.patch
git apply --index ../GenAI_Assignment_Update.patch
git diff --cached --stat
git commit -m "Organize course API and integrate Assignment 3 GAN"
git push -u origin assignment3-organized
```

The patch includes the required renames, duplicate-file removals, GAN
checkpoint, English documents, and tests. Check that `git apply --check`
succeeds before applying. It targets the inspected repository revision
`e81219b`; if the repository has changed and the check fails, reconcile the
changes instead of forcing the patch.

If you use an existing checkout, first confirm that its working tree is clean
with `git status`. Keep any of your own uncommitted changes separate before
applying the update. After the push, review the branch and merge it into
`main` when ready. Run the Docker deployment on the target computer before
claiming a successful container test.

The organized ZIP is a runnable snapshot without `.git` metadata. Uploading
that ZIP as one file to GitHub does not apply its contents as repository files.
Use a Git checkout and the patch above to preserve both the directory
structure and removals.

## If the error mentions origin

`organized` describes project structure. `origin` is a Git remote name.
Inspect remotes in the actual local checkout:

```bash
git remote -v
```

If there is no remote named `origin`, add the course repository:

```bash
git remote add origin https://github.com/ll3980/GenAI_Assignment.git
```

If `origin` exists but its URL is incorrect, update that remote instead:

```bash
git remote set-url origin https://github.com/ll3980/GenAI_Assignment.git
```

An authentication error requires signing into GitHub with an authorized
account. The exact error message determines which action applies.
