# Skapa bas-SIE-X
engine = SemanticIntelligenceEngine()

# Exempel 1: Juridiskt AI-system
legal_transformer = LegalTransformer()
legal_transformer.inject(engine)

# Nu är engine ett juridiskt AI-system!
result = await engine.extract_async(
    "Enligt 4 kap. 1 § BrB ska den som uppsåtligen berövar annan livet dömas för mord."
)
print(result['legal_entities'])  # Juridiska entiteter
print(result['applicable_law'])  # Tillämplig lag

# Exempel 2: Medicinskt diagnossystem
engine2 = SemanticIntelligenceEngine()
medical_transformer = MedicalTransformer()
medical_transformer.inject(engine2)

result = await engine2.extract_async(
    "Patienten uppvisar feber, hosta och andningssvårigheter sedan 3 dagar."
)
print(result['differential_diagnosis'])  # Möjliga diagnoser
print(result['clinical_recommendations'])  # Rekommendationer

# Exempel 3: Hybrid Financial-Legal system
engine3 = SemanticIntelligenceEngine()
loader = TransformerLoader(engine3)
loader.create_hybrid_system(['financial', 'legal'])

result = await engine3.extract_async(
    "Insiderhandel enligt 2005:377 MAR kan leda till börsfall på 15% för AAPL."
)
print(result['financial']['trading_signals'])  # Finansiella signaler
print(result['legal']['compliance'])  # Juridisk compliance
print(result['combined_insights'])  # Kombinerade insikter