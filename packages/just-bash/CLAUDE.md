# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

just-bash is a TypeScript implementation of a bash interpreter with an in-memory virtual filesystem. Designed for AI agents needing a secure bash environment. Browser-compatible, no WASM dependencies allowed.

## Commands

```bash
# Build & Lint
npm run build                 # Build TypeScript (required before using dist/)
npm run typecheck             # Type check
npm run lint:fix              # Fix lint errors (biome)
npm run knip                  # Check for unused exports/dependencies

# Testing
npm run test:run              # Run ALL tests (including spec tests)
npm run test:unit             # Run unit tests only (fast, no comparison/spec)
npm run test:comparison       # Run comparison tests only (uses fixtures)
npm run test:comparison:record # Re-record comparison test fixtures

# Run specific test file
npx vitest run src/commands/grep/grep.basic.test.ts

# Run specific spec test file by name pattern
npx vitest run src/spec-tests/spec.test.ts -t "arith.test.sh"
npx vitest run src/spec-tests/spec.test.ts -t "array-basic.test.sh"
```

## Architecture

### Core Pipeline

```
Input Script → Parser (src/parser/) → AST (src/ast/) → Interpreter (src/interpreter/) → ExecResult
```

### Key Modules

**Parser** (`src/parser/`): Recursive descent parser producing AST nodes

- `lexer.ts` - Tokenizer with bash-specific handling (heredocs, quotes, expansions)
- `parser.ts` - Main parser orchestrating specialized sub-parsers
- `expansion-parser.ts` - Parameter expansion, command substitution parsing
- `compound-parser.ts` - if/for/while/case/function parsing

**Interpreter** (`src/interpreter/`): AST execution engine

- `interpreter.ts` - Main execution loop, command dispatch
- `expansion.ts` - Word expansion (parameter, brace, glob, tilde, command substitution)
- `arithmetic.ts` - `$((...))` and `((...))` evaluation
- `conditionals.ts` - `[[ ]]` and `[ ]` test evaluation
- `control-flow.ts` - Loops and conditionals execution
- `builtins/` - Shell builtins (export, local, declare, read, etc.)

**Commands** (`src/commands/`): External command implementations

- Each command in its own directory with implementation + tests
- Registry pattern via `registry.ts`

**Filesystem** (`src/fs/`): In-memory VFS and MountableFs for combining filesystems

**AWK** (`src/commands/awk/`): AWK text processing implementation

- `parser.ts` - Parses AWK programs (BEGIN/END blocks, rules, user-defined functions)
- `executor.ts` - Executes parsed AWK programs line by line
- `expressions.ts` - Expression evaluation (arithmetic, string functions, comparisons)
- Supports: field splitting, pattern matching, printf, gsub/sub/split, user-defined functions
- Limitations: User-defined functions support single return expressions only (no multi-statement bodies or if/else)

**SED** (`src/commands/sed/`): Stream editor implementation

- `parser.ts` - Parses sed commands and addresses
- `executor.ts` - Executes sed commands with pattern/hold space
- Supports: s, d, p, q, n, a, i, c, y, =, addresses, ranges, extended regex (-E/-r)
- Has execution limits to prevent runaway compute

### Adding Commands

Commands go in `src/commands/<name>/` with:

1. Implementation file with usage statement
2. Unit tests (collocated `*.test.ts`)
3. Error on unknown options (unless real bash ignores them)
4. Comparison tests in `src/comparison-tests/` for behavior validation

### Testing Strategy

- **Unit tests**: Fast, isolated tests for specific functionality
- **Comparison tests**: Compare just-bash output against recorded bash fixtures (see `src/comparison-tests/README.md`)
- **Spec tests** (`src/spec-tests/`): Bash specification conformance (may have known failures)

Prefer comparison tests when uncertain about bash behavior. Keep test files under 300 lines.

### Comparison Tests (Fixture System)

Comparison tests use pre-recorded bash outputs stored in `src/comparison-tests/fixtures/`. This eliminates platform differences (macOS vs Linux). See `src/comparison-tests/README.md` for details.

```bash
# Run comparison tests (uses fixtures, no real bash needed)
npm run test:comparison

# Re-record fixtures (skips locked fixtures)
RECORD_FIXTURES=1 npx vitest run src/comparison-tests/mytest.comparison.test.ts

# Force re-record including locked fixtures
RECORD_FIXTURES=force npm run test:comparison
```

When adding comparison tests:
1. Write the test using `setupFiles()` and `compareOutputs()`
2. Run with `RECORD_FIXTURES=1` to generate fixtures
3. Commit both the test file and the generated fixture JSON
4. If manually adjusting for Linux behavior, add `"locked": true` to the fixture

## Development Guidelines

- Read AGENTS.md
- Always verify with `npm run typecheck && npm run lint:fix && npm run knip && npm run test:run` before finishing
- Assert full stdout/stderr in tests, not partial matches
- Implementation must match real bash behavior, not convenience
- Dependencies using WASM are not allowed
- We explicitly don't support 64-bit integers
- All parsing/execution must have reasonable limits to prevent runaway compute
