"""
Universal Transformer Loader - Dynamiskt ladda och applicera transformers
"""


class TransformerLoader:
    """
    Laddar och hanterar olika transformers
    """

    def __init__(self, sie_x_engine):
        self.engine = sie_x_engine
        self.active_transformers = []
        self.available_transformers = {
            'legal': LegalTransformer,
            'medical': MedicalTransformer,
            'financial': FinancialTransformer,
            'creative': CreativeTransformer
        }

    def load_transformer(self, transformer_type: str, config: Dict = None):
        """Ladda och aktivera en transformer."""
        if transformer_type not in self.available_transformers:
            raise ValueError(f"Unknown transformer: {transformer_type}")

        # Skapa transformer-instans
        transformer_class = self.available_transformers[transformer_type]
        transformer = transformer_class(**(config or {}))

        # Injicera i engine
        transformer.inject(self.engine)

        self.active_transformers.append({
            'type': transformer_type,
            'instance': transformer,
            'loaded_at': datetime.now()
        })

        return transformer

    def create_hybrid_system(self, transformer_types: List[str]):
        """Skapa hybridsystem med flera transformers."""
        print(f"🔧 Creating hybrid system with: {', '.join(transformer_types)}")

        for t_type in transformer_types:
            self.load_transformer(t_type)

        # Skapa hybrid extract-funktion
        original_extract = self.engine._original_extract

        async def hybrid_extract(text: str, **kwargs):
            results = {
                'base': await original_extract(text, **kwargs)
            }

            # Kör alla transformers
            for transformer_info in self.active_transformers:
                t_type = transformer_info['type']
                t_instance = transformer_info['instance']

                # Varje transformer får sin egen nyckel
                transformed_func = t_instance.transform_extraction(original_extract)
                results[t_type] = await transformed_func(text, **kwargs)

            # Kombinera insights
            results['combined_insights'] = self._combine_insights(results)
            results['cross_domain_connections'] = self._find_cross_domain_connections(results)

            return results

        self.engine.extract_async = hybrid_extract

        print(f"✨ Hybrid system created: {self.engine.__class__.__name__} + {' + '.join(transformer_types)}")