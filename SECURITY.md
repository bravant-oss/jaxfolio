# Security Policy

## Supported versions

jaxfolio is pre-1.0 and follows a "latest minor" support model: security fixes
are applied to the most recent released minor version. We recommend always
running the latest release.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

Once the project reaches 1.0, this table will be updated to a longer support
window (see [SUPPORT.md](SUPPORT.md)).

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, use GitHub's private vulnerability reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability** to open a private advisory.

This delivers your report directly to the maintainers without disclosing it
publicly. If private reporting is unavailable to you, open a minimal public issue
asking a maintainer to contact you privately — do **not** include exploit details
there.

Please include, as available:

- A description of the vulnerability and its impact.
- Steps to reproduce or a proof of concept.
- Affected version(s) and environment (OS, Python, JAX version).

### Response commitment

- **Acknowledgement:** within 3 business days.
- **Initial assessment:** within 10 business days.
- **Fix & disclosure:** we aim to release a patch and publish an advisory within
  30 days of confirmation, coordinating a disclosure timeline with the reporter.

We will credit reporters in the advisory unless you ask to remain anonymous.

## Security-relevant design notes

- **Local-only LLM integration.** The optional `llm` extra talks to a **local**
  model server (Ollama, default `http://localhost:11434`). jaxfolio does not ship
  or require any cloud API keys, and no market data, prompts, or portfolio
  contents leave your machine unless you explicitly point the client at a remote
  endpoint. Treat any custom endpoint you configure as you would any outbound
  network dependency.
- **Data loaders.** The optional `data` extra (`load_yfinance`) makes outbound
  network requests to third-party data providers. Validate and sanitize any
  externally sourced data before feeding it into a pipeline.
- **Untrusted input.** jaxfolio is a numerical/research library; it does not
  sandbox arbitrary code. Do not pass untrusted user input to `custom_strategy`
  objective closures, which execute as ordinary Python/JAX.

## Scope

jaxfolio is research/simulation software and is **not** investment advice or a
trading/execution system. See [DISCLAIMER.md](DISCLAIMER.md). Reports about the
*financial soundness of a strategy* are feature/discussion topics, not security
vulnerabilities.
