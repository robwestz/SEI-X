"""
Legal Intelligence Transformer - Transformerar SIE-X till ett juridiskt AI-system
"""

from typing import Dict, List, Any, Tuple
import re
from dataclasses import dataclass


@dataclass
class LegalEntity:
    """Juridisk entitet med speciella egenskaper."""
    type: str  # 'law', 'case', 'regulation', 'precedent'
    jurisdiction: str
    citations: List[str]
    binding_authority: bool
    temporal_validity: Tuple[str, str]  # (from_date, to_date)


class LegalTransformer:
    """
    Minimal modul som transformerar SIE-X till LegalAI-X
    """

    def __init__(self):
        self.legal_patterns = {
            'swedish_law': r'(\d{4}:\d+)',  # SFS-nummer
            'eu_regulation': r'(EU|EG)\s*\d{4}/\d+',
            'case_citation': r'(NJA|RH|HFD|AD)\s*\d{4}\s*s\.\s*\d+',
            'paragraph': r'§\s*\d+(?:\s*\w)?'
        }
        self.jurisdiction_hierarchy = {
            'EU': 1,
            'Swedish_Constitution': 2,
            'Swedish_Law': 3,
            'Government_Regulation': 4,
            'Agency_Regulation': 5
        }

    def transform_extraction(self, original_extract_func):
        """Wrapper som transformerar extraction till juridisk analys."""

        async def legal_extract(text: str, **kwargs) -> Dict[str, Any]:
            # Kör original extraktion
            keywords = await original_extract_func(text, **kwargs)

            # Transformera till juridiska entiteter
            legal_entities = []
            legal_graph = {'nodes': [], 'edges': []}

            for kw in keywords:
                # Identifiera juridiska entiteter
                legal_type = self._classify_legal_entity(kw.text)

                if legal_type:
                    entity = LegalEntity(
                        type=legal_type['type'],
                        jurisdiction=legal_type['jurisdiction'],
                        citations=self._extract_citations(kw.text),
                        binding_authority=self._check_binding_authority(kw.text),
                        temporal_validity=self._get_temporal_validity(kw.text)
                    )

                    # Skapa juridisk graf
                    legal_graph['nodes'].append({
                        'id': kw.text,
                        'type': entity.type,
                        'authority_level': self.jurisdiction_hierarchy.get(
                            entity.jurisdiction, 99
                        )
                    })

                    # Hitta juridiska relationer
                    for other_kw in keywords:
                        if other_kw != kw:
                            relation = self._find_legal_relation(kw.text, other_kw.text)
                            if relation:
                                legal_graph['edges'].append({
                                    'from': kw.text,
                                    'to': other_kw.text,
                                    'type': relation
                                })

                legal_entities.append({
                    'entity': kw.text,
                    'legal_metadata': entity,
                    'original_score': kw.score,
                    'legal_weight': self._calculate_legal_weight(entity)
                })

            # Bygg juridisk hierarki
            hierarchy = self._build_legal_hierarchy(legal_entities)

            # Identifiera konflikter
            conflicts = self._detect_legal_conflicts(legal_entities)

            return {
                'keywords': keywords,  # Behåll original
                'legal_entities': legal_entities,
                'legal_graph': legal_graph,
                'hierarchy': hierarchy,
                'conflicts': conflicts,
                'legal_summary': self._generate_legal_summary(legal_entities),
                'applicable_law': self._determine_applicable_law(legal_entities)
            }

        return legal_extract

    def _classify_legal_entity(self, text: str) -> Optional[Dict[str, str]]:
        """Klassificera juridisk entitet."""
        for pattern_name, pattern in self.legal_patterns.items():
            if re.search(pattern, text):
                return {
                    'type': pattern_name.split('_')[0],
                    'jurisdiction': 'EU' if 'eu' in pattern_name else 'Sweden'
                }
        return None

    def _calculate_legal_weight(self, entity: LegalEntity) -> float:
        """Beräkna juridisk vikt baserat på hierarki."""
        base_weight = 1.0 / self.jurisdiction_hierarchy.get(entity.jurisdiction, 99)
        if entity.binding_authority:
            base_weight *= 2
        return base_weight

    def inject(self, sie_x_engine):
        """Injicera transformer i SIE-X engine."""
        # Spara original metoder
        sie_x_engine._original_extract = sie_x_engine.extract_async

        # Ersätt med transformerade versioner
        sie_x_engine.extract_async = self.transform_extraction(
            sie_x_engine._original_extract
        )

        # Lägg till juridiska metoder
        sie_x_engine.find_applicable_law = self._determine_applicable_law
        sie_x_engine.check_legal_compliance = self._check_compliance
        sie_x_engine.generate_legal_memo = self._generate_legal_memo

        print("🏛️ SIE-X transformed into LegalAI-X")