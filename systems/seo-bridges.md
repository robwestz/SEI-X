---
name: SEO Bridge Analyzer
version: "1.0"
description: >
  Analyserar semantisk distans mellan publisher och target,
  hittar bryggtopics och genererar writer-constraints.
  Samma logik som BACOWR v5 preflight — men driven av SIE-X-motorn.

engine:
  mode: balanced
  embedding_model: paraphrase-multilingual-MiniLM-L12-v2
  min_confidence: 0.3
  top_k: 20

transformers:
  - seo

stages:
  - extract_keywords
  - find_bridges
  - assess_risk
  - generate_constraints
  - format_output
  - save_output

input:
  format: csv
  schema:
    required: [publisher_domain, target_url, anchor_text]

output:
  format: json
  directory: output/seo/

stage_config:
  extract_keywords:
    top_k: 25
    min_confidence: 0.25
  find_bridges:
    bridge_types: [strong, pivot, wrapper]
    max_bridges: 3
  assess_risk:
    ymyl_patterns: ["hälsa", "health", "medicin", "juridik", "legal", "ekonomi"]
---

# SEO Bridge Analyzer — Domänregler

## Syfte

Detta system analyserar den semantiska kopplingen mellan en **publisher-domän**
(sajten som publicerar artikeln) och en **target-URL** (kundens målsida som
ankarlänken pekar till).

Resultatet är en **preflight-analys** som ger:
1. Semantisk distans (cosine similarity)
2. Bryggtopics (pivot-ämnen som binder publisher och target)
3. Writer-constraints (regler för AI-skribenten)
4. Riskbedömning

## Hur tolkas resultatet

### Semantisk distans

| Avstånd | Tolkning | Åtgärd |
|---------|----------|--------|
| >= 0.90 (identical) | Samma ämne | Direkt koppling, exakt ankartext säker |
| >= 0.70 (close) | Närliggande | Gemensamma entiteter räcker som brygga |
| >= 0.50 (moderate) | Viss koppling | Behöver tydlig bridge via pivot-topic |
| >= 0.30 (distant) | Svag koppling | Explicit variabelgifte-strategi krävs |
| < 0.30 (unrelated) | Ingen koppling | Varning — risken att det blir onaturligt är hög |

### Bryggtopics

Tre typer:
- **Strong** — Direkt överlapp. Publisher och target delar entiteter.
- **Pivot** — Tematisk mellanlandning. Ett ämne som existerar i båda världar.
- **Wrapper** — Kontextuell inramning. En vinkel som gör kopplingen naturlig.

### Writer-constraints

JSON-struktur som skickas till AI-skribenten:

```json
{
  "required_entities": ["sportstatistik", "analys"],
  "forbidden_entities": ["spelansvar", "spelmissbruk"],
  "recommended_angle": "Hur realtidsdata förändrar sportanalys",
  "anchor_placement": "word 150-700",
  "trust_link_topics": ["statistik", "forskning"]
}
```

## Regler

1. **Bryggan avgör vinkeln.** Artikelns ämne bestäms av bryggan, inte av target.
2. **Publisher-domänen sätter taket.** Man kan inte skriva fritt om casino på en byggsajt.
3. **Kontextlänkar flyttar taket.** En bra trustlänk binder publisher och target samman.
4. **Forbidden entities är hårda.** Nämn aldrig spelansvarsorganisationer i artikeltext.
5. **YMYL-ämnen kräver auktoritativa källor.** Hälsa, ekonomi, juridik — extra försiktighet.
