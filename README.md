# Translation Automation Tool

Automated translation pipeline for mobile app localization, replacing manual workflows and eliminating a €12,000/year SaaS licensing cost.

## Context

Managing translations for a Flutter mobile app across 8 languages required either:
- Manual copy-paste into Google Translate (no context, time-consuming)
- A Lokalise AI license at €12,000/year

This tool automates the full pipeline: extract strings from Lokalise, translate via OpenAI API with product context, and push results back — in one command.

## How it works

1. Fetch untranslated strings from **Lokalise API**
2. Send to **OpenAI API** with domain-specific context (IoT energy management, UI copy)
3. Push translated strings back to **Lokalise API**
4. Ready for review and release

## Stack

- **Python** — core pipeline
- **FastAPI** — REST API wrapper
- **OpenAI API** — translation engine (GPT-4)
- **Lokalise API** — translation management platform
- **Render** — cloud deployment

## Impact

- Eliminated €12,000/year Lokalise AI license
- Reduced translation cycle from hours to minutes
- Covers 8 languages, used for every feature release (4–5/year) and new language onboarding (1–2/year)
- Deployed and used in production

## Deployment

Live at: `https://translation-auto.onrender.com/`

## Author

Built autonomously as a PM-driven engineering initiative to replace external tooling with a fit-for-purpose internal solution.
