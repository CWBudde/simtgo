# Engineering notes

The durable findings: what was measured, what broke, and why things are the way
they are. These pages are filed by **subject**, so a finding stays findable
after the work that produced it is forgotten.

| Page                                     | What it holds                                                      |
| ---------------------------------------- | ------------------------------------------------------------------ |
| [decisions.md](decisions.md)             | settled questions and the reasoning that settled them              |
| [emitter-defects.md](emitter-defects.md) | every mistranslation found, its mechanism, and what catches it now |
| [verification.md](verification.md)       | the layers of checking, and what each one cannot see               |
| [toolchain.md](toolchain.md)             | measured behaviour of the driver, NVRTC and PTX                    |
| [tile.md](tile.md)                       | the tile track: fusion, halo staging, aliasing, and where it stops |

## What does not live here

| Question                                  | Answer                                                                        |
| ----------------------------------------- | ----------------------------------------------------------------------------- |
| What does the subset accept and refuse?   | [`../SPEC.md`](../SPEC.md) — the contract, checked against the implementation |
| What may a test assert about a `float32`? | [`../NUMERICS.md`](../NUMERICS.md)                                            |
| What is still to be done?                 | [`../PLAN.md`](../PLAN.md)                                                    |
| What is this, and how do I run it?        | [`../README.md`](../README.md)                                                |
| How do I work in this repository?         | [`../CLAUDE.md`](../CLAUDE.md)                                                |

A claim here should not contradict one of those. Where the same fact appears
twice, the file in the table above owns it and these pages link to it.
