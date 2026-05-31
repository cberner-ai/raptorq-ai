# Agent instructions for raptorq-ai

This file tells coding agents how to work productively in this repository.

## Before completing codebase-changing work

Run `just test` and confirm it passes after making any change that can affect the codebase
in the current working directory.

This target runs the `pre` recipe first, which executes the protected Rust file check,
`cargo deny --all-features check licenses`, `cargo fmt --all -- --check`, and
`cargo clippy --all --all-targets -- -Dwarnings`, and then runs the Rust build and test suite.
If any of those fail, fix the underlying issue; do not bypass checks.

## Style guide

- Comments should be brief and focus on important invariants, architectural details, or other
  long-term relevant information. They should not contain minor implementation details of the
  current commit.

## Tests

When adding new features, add tests, but aim for high code coverage and important integration
tests without adding too many lines of new test code. 90% coverage is a good target for new
features; it does not have to be 100%. Expanding a logically related existing test is often a good
way to achieve coverage without bloating the suite.

## Git commits

1. Git commits should use your human's name and email address for authorship. Add `Assisted-by:`
   and your agent name at the end of the commit message, in the style of the Linux kernel's coding
   assistant guidelines.
2. Make one commit per feature or bug fix when opening a PR. Multiple commits or fixup commits
   should not be merged to master.
