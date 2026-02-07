---
name: Legal Compliance Reviewer
version: "1.0"
description: >
  Granskar juridiska texter: identifierar lagrum, kontrollerar
  jurisdiktionshierarki, hittar konflikter och genererar legal memo.

engine:
  mode: balanced
  min_confidence: 0.35
  top_k: 25

transformers:
  - legal

stages:
  - extract_keywords
  - extract_legal_entities
  - legal_compliance_check
  - format_output
  - save_output

input:
  format: text
  schema:
    description: "Contract, regulation, or legal document in Swedish or English"

output:
  format: markdown
  directory: output/legal/
---

# Legal Compliance Reviewer — Domänregler

## Syfte

Analyserar juridiska texter och identifierar:
1. Lagrumshänvisningar (SFS, EU-förordningar, rättsfall)
2. Jurisdiktionshierarki och auktoritetsnivå
3. Potentiella konflikter mellan rättskällor
4. Compliance-status mot specificerade regelverk

## Hierarki

```
EU-rätt (förordningar, direktiv)
  └── Grundlag (RF, TF, YGL, SO)
       └── Svensk lag (SFS YYYY:NNN)
            └── Förordning
                 └── Myndighetsföreskrift (HSLF-FS, SOSFS, etc.)
```

Högre nivå trumfar alltid lägre. Vid konflikt: flagga och ange vilken källa som har företräde.

## Riskklassificering

| Nivå | Trigger | Åtgärd |
|------|---------|--------|
| CRITICAL | Grundlagskonflikt, GDPR-brott | Omedelbar flaggning |
| HIGH | Lagkonflikt, compliance-gap | Kräver juridisk granskning |
| MEDIUM | Otydligt rättsläge, tolkningsfråga | Rekommendation |
| LOW | Formellt korrekt, mindre anmärkning | Notering |
