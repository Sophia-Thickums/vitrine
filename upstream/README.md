# The upstream folder — what is here, and what has not happened

**Nothing here has been sent to Citron Neo.** These are the two artifacts the first-contact
conversation needs, prepared so that conversation is one paste rather than a session of writing.

| file | what it is |
|---|---|
| `OFFER-20261029.md` | the case for the crate, in a maintainer's register: what it does, what is measured and with what instrument, how it plugs in, and what it does not do |
| `0001-registered-cache-highest-version.patch` | the fallback shape — the version-selection defect fixed in C++, if they would rather have a patch than a crate |

## The gate, quoted

Citron Neo's `README.md` at `c3aa70f`, under `## Contributing`:

> Issues and pull requests are welcome. Ask on Discord before starting something

So the first contact is a conversation on their Discord. That is the one step in this folder that is
not mine to take, and it is not a formality: it is the project asking to be met before being changed.

## After the conversation

- **If they want the crate**: open an issue linking the repo and `OFFER-20261029.md` as the body, and
  offer the pipeline cache module first — it is the piece with a measured argument and no dependency
  on a working CPU.
- **If they want C++ changes**: `git am upstream/0001-registered-cache-highest-version.patch` in a
  clone, push a branch, open the PR. The patch states in its own message that it was not compiled,
  because this machine has neither Qt, Boost nor the dynarmic submodule.
- **If they want neither**: say so on the issue and close it. A crate nobody asked for is a repo, not
  a contribution, and the repo stands on its own either way.

## What must not be claimed

The offer is not a promise that a game runs on this. No LL/SC, no SIMD/NEON, no AArch32, no MMU in the
JIT's memory model, and the backend has never been compiled inside their tree.
