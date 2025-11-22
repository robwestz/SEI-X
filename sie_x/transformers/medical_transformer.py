"""
Medical Diagnostics Transformer - Transforms SIE-X into MedicalAI-X
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MedicalCode:
    """Medical coding representation."""
    system: str  # ICD-11, SNOMED-CT, RxNorm
    code: str
    description: str
    confidence: float = 1.0


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
        """Inject medical transformer into SIE-X engine."""
        sie_x_engine._original_extract = sie_x_engine.extract_async
        sie_x_engine.extract_async = self.transform_extraction(
            sie_x_engine._original_extract
        )

        # Add medical methods
        sie_x_engine.diagnose = self._differential_diagnosis
        sie_x_engine.check_drug_safety = self._check_drug_interactions
        sie_x_engine.generate_soap_note = self._generate_soap_note
        sie_x_engine.calculate_risk_scores = self._calculate_risk_scores

        logger.info("🏥 SIE-X transformed into MedicalAI-X")

    # ========== Medical Ontology Loaders ==========

    def _load_icd11(self) -> Dict[str, str]:
        """Load ICD-11 medical codes (simplified subset)."""
        # In production, this would load from actual ICD-11 database
        return {
            'diabetes': 'ICD-11:5A10',
            'hypertension': 'ICD-11:BA00',
            'covid-19': 'ICD-11:RA01',
            'pneumonia': 'ICD-11:CA40',
            'asthma': 'ICD-11:CA23',
            'depression': 'ICD-11:6A70',
            'anxiety': 'ICD-11:6B00',
            'heart failure': 'ICD-11:BD10',
            'stroke': 'ICD-11:8B20',
            'cancer': 'ICD-11:2A00',
        }

    def _load_snomed(self) -> Dict[str, str]:
        """Load SNOMED-CT clinical terminology (simplified subset)."""
        # In production, this would load from actual SNOMED-CT database
        return {
            'fever': 'SNOMED:386661006',
            'cough': 'SNOMED:49727002',
            'headache': 'SNOMED:25064002',
            'fatigue': 'SNOMED:84229001',
            'shortness of breath': 'SNOMED:267036007',
            'chest pain': 'SNOMED:29857009',
            'nausea': 'SNOMED:422587007',
            'vomiting': 'SNOMED:422400008',
            'dizziness': 'SNOMED:404640003',
            'pain': 'SNOMED:22253000',
        }

    def _load_rxnorm(self) -> Dict[str, str]:
        """Load RxNorm medication codes (simplified subset)."""
        # In production, this would load from actual RxNorm database
        return {
            'aspirin': 'RxNorm:1191',
            'ibuprofen': 'RxNorm:5640',
            'metformin': 'RxNorm:6809',
            'lisinopril': 'RxNorm:29046',
            'atorvastatin': 'RxNorm:83367',
            'amoxicillin': 'RxNorm:723',
            'omeprazole': 'RxNorm:7646',
            'amlodipine': 'RxNorm:17767',
            'metoprolol': 'RxNorm:6918',
            'losartan': 'RxNorm:52175',
        }

    # ========== Entity Classification ==========

    def _classify_medical_entity(self, text: str) -> Tuple[Optional[str], List[MedicalCode]]:
        """Classify medical entity and return codes."""
        text_lower = text.lower()
        codes = []

        # Check symptoms
        for symptom, snomed_code in self.medical_ontologies['SNOMED-CT'].items():
            if symptom in text_lower:
                codes.append(MedicalCode('SNOMED-CT', snomed_code, symptom, 0.9))
                return 'symptoms', codes

        # Check conditions
        for condition, icd_code in self.medical_ontologies['ICD-11'].items():
            if condition in text_lower:
                codes.append(MedicalCode('ICD-11', icd_code, condition, 0.85))
                return 'conditions', codes

        # Check medications
        for med, rxnorm_code in self.medical_ontologies['RxNorm'].items():
            if med in text_lower:
                codes.append(MedicalCode('RxNorm', rxnorm_code, med, 0.95))
                return 'medications', codes

        # Check for lab values (basic pattern matching)
        if re.search(r'\d+\s*(mg/dl|mmol/l|%|bpm)', text_lower):
            return 'lab_values', []

        # Check for procedures
        procedure_keywords = ['surgery', 'procedure', 'operation', 'biopsy', 'scan', 'mri', 'ct', 'x-ray']
        if any(kw in text_lower for kw in procedure_keywords):
            return 'procedures', []

        return None, []

    def _assess_severity(self, text: str, score: float) -> str:
        """Assess severity of medical entity."""
        text_lower = text.lower()

        # Critical symptoms
        critical_symptoms = ['chest pain', 'stroke', 'heart attack', 'severe bleeding', 'unconscious']
        if any(symptom in text_lower for symptom in critical_symptoms):
            return 'critical'

        # High severity
        high_severity = ['severe', 'acute', 'emergency', 'urgent']
        if any(word in text_lower for word in high_severity) or score > 0.8:
            return 'high'

        # Moderate severity
        moderate = ['moderate', 'persistent', 'chronic']
        if any(word in text_lower for word in moderate) or score > 0.5:
            return 'moderate'

        return 'low'

    def _extract_temporal_info(self, text: str) -> Dict[str, Any]:
        """Extract temporal information from medical text."""
        text_lower = text.lower()

        temporal_info = {
            'onset': None,
            'duration': None,
            'frequency': None
        }

        # Onset patterns
        if 'sudden' in text_lower or 'acute' in text_lower:
            temporal_info['onset'] = 'sudden'
        elif 'gradual' in text_lower or 'slowly' in text_lower:
            temporal_info['onset'] = 'gradual'

        # Duration patterns
        duration_pattern = r'(\d+)\s*(day|week|month|year|hour)s?'
        duration_match = re.search(duration_pattern, text_lower)
        if duration_match:
            temporal_info['duration'] = f"{duration_match.group(1)} {duration_match.group(2)}s"

        # Frequency patterns
        if 'constant' in text_lower or 'continuous' in text_lower:
            temporal_info['frequency'] = 'constant'
        elif 'intermittent' in text_lower or 'occasionally' in text_lower:
            temporal_info['frequency'] = 'intermittent'

        return temporal_info

    def _check_negation(self, keyword: str, full_text: str) -> bool:
        """Check if medical entity is negated in context."""
        # Find keyword position in text
        keyword_lower = keyword.lower()
        text_lower = full_text.lower()

        keyword_pos = text_lower.find(keyword_lower)
        if keyword_pos == -1:
            return False

        # Check preceding words for negation
        preceding_text = text_lower[max(0, keyword_pos - 50):keyword_pos]
        negation_words = ['no', 'not', 'without', 'denies', 'negative', 'absent', 'never']

        return any(neg in preceding_text.split() for neg in negation_words)

    # ========== Drug Interactions ==========

    def _check_drug_interactions(self, medications: List[Dict]) -> List[Dict]:
        """Check for drug-drug interactions."""
        interactions = []

        # Simple interaction database (in production, use comprehensive database)
        known_interactions = {
            ('aspirin', 'ibuprofen'): 'Increased bleeding risk',
            ('metformin', 'alcohol'): 'Risk of lactic acidosis',
            ('atorvastatin', 'grapefruit'): 'Increased statin levels',
            ('lisinopril', 'potassium'): 'Hyperkalemia risk',
        }

        med_names = [med['text'].lower() for med in medications]

        for (drug1, drug2), interaction in known_interactions.items():
            if drug1 in med_names and drug2 in med_names:
                interactions.append({
                    'drugs': [drug1, drug2],
                    'interaction': interaction,
                    'severity': 'moderate',
                    'recommendation': f'Monitor patient closely for {interaction.lower()}'
                })

        return interactions

    # ========== Clinical Recommendations ==========

    def _generate_clinical_recommendations(
            self,
            medical_entities: Dict[str, List],
            differential: List[Dict]
    ) -> List[str]:
        """Generate clinical recommendations based on findings."""
        recommendations = []

        # Check for symptoms requiring urgent care
        symptoms = medical_entities.get('symptoms', [])
        urgent_symptoms = [s for s in symptoms if s.get('severity') == 'critical']
        if urgent_symptoms:
            recommendations.append('Urgent: Immediate medical evaluation required')

        # Medication recommendations
        medications = medical_entities.get('medications', [])
        if len(medications) > 5:
            recommendations.append('Consider medication reconciliation - patient on multiple medications')

        # Diagnostic recommendations based on differential
        if differential:
            top_diagnosis = differential[0]
            recommendations.append(
                f"Consider {top_diagnosis['condition']} (probability: {top_diagnosis['probability']:.1%})"
            )

        # Risk factor recommendations
        risk_factors = medical_entities.get('risk_factors', [])
        if risk_factors:
            recommendations.append('Address identified risk factors through lifestyle modification')

        return recommendations

    def _calculate_risk_scores(
            self,
            medical_entities: Dict[str, List],
            patient_history: Optional[Dict]
    ) -> Dict[str, float]:
        """Calculate various clinical risk scores."""
        scores = {}

        # Example: Cardiovascular risk score
        cv_risk_factors = 0
        conditions = medical_entities.get('conditions', [])
        condition_names = [c['text'].lower() for c in conditions]

        if 'hypertension' in condition_names:
            cv_risk_factors += 1
        if 'diabetes' in condition_names:
            cv_risk_factors += 1
        if patient_history and patient_history.get('smoking'):
            cv_risk_factors += 1

        scores['cardiovascular_risk'] = min(cv_risk_factors / 5.0, 1.0)

        # Example: Sepsis risk
        symptoms = medical_entities.get('symptoms', [])
        sepsis_indicators = ['fever', 'tachycardia', 'hypotension']
        sepsis_score = sum(1 for s in symptoms if s['text'].lower() in sepsis_indicators)
        scores['sepsis_risk'] = min(sepsis_score / 3.0, 1.0)

        return scores

    def _check_red_flags(self, medical_entities: Dict[str, List]) -> bool:
        """Check for red flag symptoms requiring immediate attention."""
        red_flags = [
            'chest pain',
            'stroke',
            'severe headache',
            'difficulty breathing',
            'loss of consciousness',
            'severe bleeding',
            'sudden vision loss'
        ]

        symptoms = medical_entities.get('symptoms', [])
        for symptom in symptoms:
            if any(flag in symptom['text'].lower() for flag in red_flags):
                return True

        return False

    def _suggest_diagnostic_tests(self, differential: List[Dict]) -> List[str]:
        """Suggest diagnostic tests based on differential diagnosis."""
        if not differential:
            return []

        tests = []
        top_conditions = [d['condition'] for d in differential[:3]]

        # Map conditions to common tests
        test_mapping = {
            'diabetes': ['HbA1c', 'Fasting glucose', 'Oral glucose tolerance test'],
            'hypertension': ['Blood pressure monitoring', 'ECG', 'Echocardiogram'],
            'pneumonia': ['Chest X-ray', 'Complete blood count', 'Sputum culture'],
            'heart failure': ['BNP', 'Echocardiogram', 'Chest X-ray'],
            'stroke': ['CT head', 'MRI brain', 'Carotid ultrasound'],
        }

        for condition in top_conditions:
            if condition in test_mapping:
                tests.extend(test_mapping[condition])

        return list(set(tests))  # Remove duplicates

    def _generate_clinical_summary(self, medical_entities: Dict[str, List]) -> str:
        """Generate a clinical summary of findings."""
        summary_parts = []

        if medical_entities.get('symptoms'):
            symptom_list = ', '.join([s['text'] for s in medical_entities['symptoms'][:5]])
            summary_parts.append(f"Presenting symptoms: {symptom_list}")

        if medical_entities.get('conditions'):
            condition_list = ', '.join([c['text'] for c in medical_entities['conditions']])
            summary_parts.append(f"Known conditions: {condition_list}")

        if medical_entities.get('medications'):
            med_count = len(medical_entities['medications'])
            summary_parts.append(f"Currently on {med_count} medication(s)")

        return '. '.join(summary_parts) if summary_parts else "No significant clinical findings documented"

    # ========== Disease Probability Calculations ==========

    def _calculate_disease_probability(
            self,
            symptoms: List[Dict],
            disease_symptoms: List[str],
            history: Optional[Dict]
    ) -> float:
        """Calculate Bayesian probability of disease given symptoms."""
        if not symptoms:
            return 0.0

        symptom_texts = [s['text'].lower() for s in symptoms]

        # Count matching symptoms
        matches = sum(1 for ds in disease_symptoms if any(ds.lower() in st for st in symptom_texts))

        # Base probability
        base_prob = matches / len(disease_symptoms) if disease_symptoms else 0.0

        # Adjust for patient history
        if history:
            # Increase probability if family history
            if history.get('family_history', {}).get(disease_symptoms[0]):
                base_prob *= 1.5

            # Increase probability if risk factors present
            if history.get('risk_factors'):
                base_prob *= 1.2

        return min(base_prob, 1.0)

    def _get_supporting_symptoms(
            self,
            patient_symptoms: List[Dict],
            disease_symptoms: List[str]
    ) -> List[str]:
        """Get symptoms that support the diagnosis."""
        patient_symptom_texts = [s['text'].lower() for s in patient_symptoms]
        return [ds for ds in disease_symptoms if any(ds.lower() in pst for pst in patient_symptom_texts)]

    def _get_missing_symptoms(
            self,
            patient_symptoms: List[Dict],
            disease_symptoms: List[str]
    ) -> List[str]:
        """Get symptoms missing from the disease profile."""
        patient_symptom_texts = [s['text'].lower() for s in patient_symptoms]
        return [ds for ds in disease_symptoms if not any(ds.lower() in pst for pst in patient_symptom_texts)]

    # ========== SOAP Note Generation ==========

    def _generate_soap_note(
            self,
            medical_entities: Dict[str, List],
            differential: List[Dict],
            patient_history: Optional[Dict] = None
    ) -> str:
        """Generate SOAP (Subjective, Objective, Assessment, Plan) note."""
        soap = []

        # Subjective
        symptoms = medical_entities.get('symptoms', [])
        if symptoms:
            soap.append("SUBJECTIVE:")
            soap.append(f"  Patient reports: {', '.join([s['text'] for s in symptoms[:5]])}")
        else:
            soap.append("SUBJECTIVE:\n  No symptoms documented")

        # Objective
        lab_values = medical_entities.get('lab_values', [])
        soap.append("\nOBJECTIVE:")
        if lab_values:
            soap.append(f"  Lab findings: {', '.join([l['text'] for l in lab_values])}")
        else:
            soap.append("  Physical examination findings pending")

        # Assessment
        soap.append("\nASSESSMENT:")
        if differential:
            for i, dx in enumerate(differential[:3], 1):
                soap.append(f"  {i}. {dx['condition']} (probability: {dx['probability']:.1%})")
        else:
            soap.append("  Assessment pending further evaluation")

        # Plan
        soap.append("\nPLAN:")
        recommendations = self._generate_clinical_recommendations(medical_entities, differential)
        for rec in recommendations:
            soap.append(f"  - {rec}")

        tests = self._suggest_diagnostic_tests(differential)
        if tests:
            soap.append(f"  - Diagnostic tests: {', '.join(tests[:3])}")

        return '\n'.join(soap)