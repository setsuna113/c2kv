# ToolSandbox deployment patches

Upstream pin: **apple/ToolSandbox @ 165848b9a78cead7ca7fe7c89c688b58e6501219**
(the server checkout's `165848b init` is this commit imported into a local
repo; its only meaningful local edits are superseded by this patch).

`0001-openai-base-url-env.patch` makes the repo reproducible for the bench
(audit BLOCKER: the TS column previously depended on a server-side edit
that existed nowhere in this repository):

* `openai_api_agent.py` — the agent client reads `OPENAI_BASE_URL`
  (upstream hard-codes `https://api.openai.com/v1` and a comment saying it
  deliberately IGNORES the env var, so a vanilla clone can produce no TS
  numbers for any arm).
* `openai_api_user.py` — the USER SIMULATOR reads its own
  `TOOLSANDBOX_USER_BASE_URL` (falling back to `OPENAI_BASE_URL` for
  standalone use).  Both roles reading the same variable routed the
  simulator through the arm proxy, making every historical TS number an
  agent+user joint degradation.  `benchmarks/run.py` exports the raw
  upstream endpoint here (`--user-upstream`).

Apply from the ToolSandbox checkout root:
`patch -p1 < benchmarks/toolsandbox_patches/0001-openai-base-url-env.patch`

TS test-mode's "n=3" is ONE base scenario (`send_message_with_contact_
content_cellular_off`) plus two perturbations (distractor tools / scrambled
arg descriptions) — not three independent tasks; only the full suite is a
real column (audit finding).
