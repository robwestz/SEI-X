"""
Medical Diagnostics Transformer - Transformerar SIE-X till MedicalAI-X
"""


class MedicalTransformer:
    """
    Transformerar SIE-X till ett medicinskt diagnossystem
    """

    def __init__(self):
        self.medical_ontologies = {
            'ICD-11': self._load_icd11(),
            'SNOMED-CT': self._load_snomed(),
            'RxNorm': self._load_rxnorm()
        }
        self.symptom_disease_map = {}
        self.drug_interaction_db = {}
        self.clinical_guidelines = {}

    def transform_extraction(self, original_extract_func):
        """Transformera extraktion till medicinsk analys."""

        async def medical_extract(text: str, patient_history: Dict = None, **kwargs):
            # Original extraktion
            keywords = await original_extract_func(text, **kwargs)

            # Medicinsk transformation
            medical_entities = {
                'symptoms': [],
                'conditions': [],
                'medications': [],
                'procedures': [],
                'lab_values': [],
                'risk_factors': []
            }

            for kw in keywords:
                # Klassificera medicinska entiteter
                med_type, med_codes = self._classify_medical_entity(kw.text)

                if med_type:
                    entity = {
                        'text': kw.text,
                        'type': med_type,
                        'codes': med_codes,
                        'severity': self._assess_severity(kw.text, kw.score),
                        'temporal': self._extract_temporal_info(kw.text),
                        'negation': self._check_negation(kw.text, text)
                    }

                    medical_entities[med_type].append(entity)

            # Kör differentialdiagnos
            differential = self._differential_diagnosis(
                medical_entities['symptoms'],
                patient_history
            )

            # Kontrollera läkemedelsinteraktioner
            interactions = self._check_drug_interactions(
                medical_entities['medications']
            )

            # Generera kliniska rekommendationer
            recommendations = self._generate_clinical_recommendations(
                medical_entities,
                differential
            )

            # Risk scoring
            risk_scores = self._calculate_risk_scores(
                medical_entities,
                patient_history
            )

            return {
                'original_keywords': keywords,
                'medical_entities': medical_entities,
                'differential_diagnosis': differential,
                'drug_interactions': interactions,
                'clinical_recommendations': recommendations,
                'risk_scores': risk_scores,
                'requires_immediate_attention': self._check_red_flags(medical_entities),
                'suggested_tests': self._suggest_diagnostic_tests(differential),
                'clinical_summary': self._generate_clinical_summary(medical_entities)
            }

        return medical_extract

    def _differential_diagnosis(self, symptoms: List[Dict], history: Dict) -> List[Dict]:
        """Generera differentialdiagnos baserat på symptom."""
        diagnoses = []

        # Använd Bayesian reasoning
        for disease, disease_symptoms in self.symptom_disease_map.items():
            probability = self._calculate_disease_probability(
                symptoms,
                disease_symptoms,
                history
            )

            if probability > 0.1:  # Threshold
                diagnoses.append({
                    'condition': disease,
                    'probability': probability,
                    'supporting_symptoms': self._get_supporting_symptoms(
                        symptoms, disease_symptoms
                    ),
                    'missing_symptoms': self._get_missing_symptoms(
                        symptoms, disease_symptoms
                    )
                })

        return sorted(diagnoses, key=lambda x: x['probability'], reverse=True)[:10]

    def inject(self, sie_x_engine):
        """Injicera medical transformer."""
        sie_x_engine._original_extract = sie_x_engine.extract_async
        sie_x_engine.extract_async = self.transform_extraction(
            sie_x_engine._original_extract
        )

        # Lägg till medicinska metoder
        sie_x_engine.diagnose = self._differential_diagnosis
        sie_x_engine.check_drug_safety = self._check_drug_interactions
        sie_x_engine.generate_soap_note = self._generate_soap_note
        sie_x_engine.calculate_risk_scores = self._calculate_risk_scores

        print("🏥 SIE-X transformed into MedicalAI-X")