---
name: Clinical Diagnostic Assistant
version: "1.0"
description: >
  Extraherar medicinska entiteter från kliniska anteckningar,
  kör differentialdiagnos och genererar SOAP-notes.
  Samma SIE-X-motor, medicinsk personlighet.

engine:
  mode: advanced
  embedding_model: all-mpnet-base-v2
  min_confidence: 0.4
  top_k: 30

transformers:
  - medical

stages:
  - extract_keywords
  - extract_medical_entities
  - differential_diagnosis
  - drug_interactions
  - generate_soap_note
  - format_output
  - save_output

input:
  format: text
  schema:
    description: "Free-text clinical note, admission note, or symptom list"

output:
  format: markdown
  directory: output/medical/

stage_config:
  extract_keywords:
    top_k: 40
    min_confidence: 0.2
  differential_diagnosis:
    max_diagnoses: 5
    include_rare: false
  drug_interactions:
    severity_threshold: moderate
---

# Clinical Diagnostic Assistant — Domänregler

## Syfte

Detta system tar emot kliniska anteckningar (fritext) och producerar:
1. Strukturerade medicinska entiteter (symptom, tillstånd, läkemedel, labvärden)
2. Differentialdiagnos med sannolikheter (Bayesiansk)
3. Läkemedelsinteraktionskontroll
4. Komplett SOAP-anteckning

## Viktiga regler

### Säkerhet
- **Red flags detekteras automatiskt**: bröstsmärta, stroke-symptom, medvetslöshet
- **Negationer respekteras**: "denies pain" → symptom INTE närvarande
- **Osäkerhet kommuniceras**: Alla diagnoser har confidence scores (0.0-1.0)
- **Systemet ersätter ALDRIG kliniskt omdöme** — det är ett beslutsstöd

### Entitetsklassificering

| Kategori | Exempel | Ontologi |
|----------|---------|----------|
| Symptom | feber, bröstsmärta, yrsel | SNOMED-CT |
| Tillstånd | diabetes, hypertension | ICD-11 |
| Läkemedel | metformin, atorvastatin | RxNorm |
| Labvärden | HbA1c 8.2%, kreatinin 120 | LOINC |
| Procedurer | appendektomi, CT thorax | CPT |
| Riskfaktorer | rökning, hereditet | — |

### Differentialdiagnos

Bayesiansk resonering:
- Priorsannolikheter baserade på prevalens
- Uppdateras med varje symptom (likelihood ratios)
- Patienthistorik justerar priors (t.ex. diabetes ökar risk för neuropati)
- Resultat sorteras efter posteriorsannolikhet

### SOAP-note format

```
SUBJECTIVE:
  Chief complaint, HPI, associated symptoms, negated symptoms

OBJECTIVE:
  Vital signs, physical exam findings, lab results

ASSESSMENT:
  1. [Diagnos 1] (p=0.65) — motivering
  2. [Diagnos 2] (p=0.20) — motivering
  3. [Diagnos 3] (p=0.10) — motivering
  Red flags: [lista eller "none"]

PLAN:
  Recommended tests, referrals, follow-up
  Drug interaction warnings (if any)
```

### Temporalitet

Systemet extraherar tidsmarkörer:
- **Onset**: sudden vs gradual
- **Duration**: akut (<48h), subakut (48h-2v), kronisk (>2v)
- **Frekvens**: konstant, intermittent, episodisk
- **Progression**: stabil, förvärras, förbättras

## Användningsmönster

### Klinisk anteckning → Strukturerad output

```
Input:  "67-årig man med bröstsmärta sedan igår morgon.
         Utstrålning vänster arm. Tidigare MI 2019.
         Tar metoprolol 50mg och ASA. Ingen dyspné."

Output: Differentialdiagnos, riskpoäng, SOAP-note
```

### Läkemedelskontroll

```
Input:  "Patient tar warfarin, ASA och ibuprofen"

Output: VARNING — warfarin + ibuprofen = ökad blödningsrisk (ALLVARLIG)
```
