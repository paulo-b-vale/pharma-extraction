"""High-level query interface for pharmaceutical knowledge base."""

import logging
from typing import Optional, Dict, Any, List
from pharma_extraction.database.mongodb_client import MongoDBClient
from pharma_extraction.database.document_store import DocumentStore
from pharma_extraction.database.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


class PharmaQueryInterface:
    """High-level query interface for pharmaceutical knowledge base.

    This class provides simple, intuitive methods for querying
    pharmaceutical data without needing to know MongoDB details.
    """

    def __init__(self, connection_string: Optional[str] = None):
        """Initialize query interface.

        Args:
            connection_string: MongoDB connection URI
        """
        self.client = MongoDBClient(connection_string)
        self.client.connect()

        self.doc_store = DocumentStore(self.client)
        self.knowledge_store = KnowledgeStore(self.client)

    def setup_database(self):
        """Setup database with indexes (run once)."""
        self.client.create_indexes()
        logger.info("Database setup complete")

    # Drug Information Queries

    def get_drug_info(self, drug_name: str) -> Dict[str, Any]:
        """Get all information about a drug.

        Args:
            drug_name: Name of the drug

        Returns:
            Dictionary with drug information

        Example:
            >>> query = PharmaQueryInterface()
            >>> info = query.get_drug_info("Paracetamol")
            >>> print(info['dosages'])
        """
        triples = self.knowledge_store.search_by_entity(drug_name)

        # Organize by relation type
        info = {
            'drug_name': drug_name,
            'dosages': [],
            'indications': [],
            'contraindications': [],
            'side_effects': [],
            'precautions': [],
            'other': []
        }

        for triple in triples:
            relation = triple.get('relation', '')
            value = triple.get('value', '')

            if 'dosage' in relation or 'dose' in relation:
                info['dosages'].append(value)
            elif 'indicado' in relation or 'indication' in relation:
                info['indications'].append(value)
            elif 'contraindicado' in relation or 'contraindication' in relation:
                info['contraindications'].append(value)
            elif 'efeito' in relation or 'side_effect' in relation:
                info['side_effects'].append(value)
            elif 'precauç' in relation or 'precaution' in relation:
                info['precautions'].append(value)
            else:
                info['other'].append({'relation': relation, 'value': value})

        return info

    def get_dosage_info(self, drug_name: str) -> List[str]:
        """Get dosage information for a drug.

        Args:
            drug_name: Name of the drug

        Returns:
            List of dosage information
        """
        relationships = self.knowledge_store.find_relationships(
            drug_name,
            relation='has_dosage'
        )
        return [value for _, _, value in relationships]

    def get_indications(self, drug_name: str) -> List[str]:
        """Get indications (uses) for a drug.

        Args:
            drug_name: Name of the drug

        Returns:
            List of indications
        """
        triples = self.knowledge_store.search_by_entity(drug_name)
        indications = [
            t['value'] for t in triples
            if 'indicado' in t.get('relation', '').lower()
        ]
        return indications

    def get_contraindications(self, drug_name: str) -> List[str]:
        """Get contraindications for a drug.

        Args:
            drug_name: Name of the drug

        Returns:
            List of contraindications
        """
        triples = self.knowledge_store.search_by_entity(drug_name)
        contraindications = [
            t['value'] for t in triples
            if 'contraindicado' in t.get('relation', '').lower()
        ]
        return contraindications

    # Document Queries

    def list_all_drugs(self) -> List[str]:
        """Get list of all drugs in the database.

        Returns:
            List of drug names
        """
        return self.knowledge_store.get_unique_entities()

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search for documents.

        Args:
            query: Search query

        Returns:
            List of matching documents
        """
        return self.doc_store.search_documents(query)

    def get_document_by_drug(self, drug_name: str) -> Optional[Dict[str, Any]]:
        """Find document for a specific drug.

        Args:
            drug_name: Name of the drug

        Returns:
            Document or None if not found
        """
        docs = self.doc_store.search_documents(drug_name, limit=1)
        return docs[0] if docs else None

    # Knowledge Graph Queries

    def find_similar_drugs(self, drug_name: str, by: str = 'indication') -> List[str]:
        """Find drugs with similar properties.

        Args:
            drug_name: Reference drug name
            by: Similarity criteria ('indication', 'side_effect', etc.)

        Returns:
            List of similar drug names

        Example:
            >>> similar = query.find_similar_drugs("Paracetamol", by="indication")
        """
        # Get target drug's properties
        target_triples = self.knowledge_store.search_by_entity(drug_name)
        target_values = set()

        for triple in target_triples:
            if by in triple.get('relation', '').lower():
                target_values.add(triple['value'])

        if not target_values:
            return []

        # Find other drugs with same properties
        similar_drugs = set()

        for value in target_values:
            triples = self.knowledge_store.search_knowledge(value)
            for triple in triples:
                entity = triple.get('entity', '')
                if entity and entity.lower() != drug_name.lower():
                    similar_drugs.add(entity)

        return list(similar_drugs)

    def search_by_symptom(self, symptom: str) -> List[str]:
        """Find drugs indicated for a symptom.

        Args:
            symptom: Symptom or condition

        Returns:
            List of drug names
        """
        triples = self.knowledge_store.search_knowledge(symptom, search_in='value')

        drugs = set()
        for triple in triples:
            if 'indicado' in triple.get('relation', '').lower():
                drugs.add(triple['entity'])

        return list(drugs)

    # Statistics and Analytics

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics.

        Returns:
            Dictionary with all statistics
        """
        return {
            'documents': self.doc_store.get_statistics(),
            'knowledge_graph': self.knowledge_store.get_statistics(),
            'database': self.client.get_database_stats()
        }

    def get_drug_stats(self, drug_name: str) -> Dict[str, Any]:
        """Get statistics for a specific drug.

        Args:
            drug_name: Name of the drug

        Returns:
            Dictionary with drug statistics
        """
        return self.knowledge_store.get_entity_stats(drug_name)

    # Advanced Queries

    def query_by_section(
        self,
        section_title: str,
        drug_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query information from specific document section.

        Args:
            section_title: Section title (e.g., "POSOLOGIA")
            drug_name: Optional drug name filter

        Returns:
            List of phrases from that section
        """
        # Search phrases by section
        docs = self.doc_store.search_documents(drug_name) if drug_name else None
        doc_ids = [doc['_id'] for doc in docs] if docs else None

        all_triples = []
        if doc_ids:
            for doc_id in doc_ids:
                triples = self.knowledge_store.get_triples_by_document(doc_id)
                all_triples.extend([
                    t for t in triples
                    if section_title.lower() in t.get('context', {}).get('section_title', '').lower()
                ])
        else:
            # Search all triples
            all_triples = self.knowledge_store.triples.find({
                'context.section_title': {'$regex': section_title, '$options': 'i'}
            })

        return list(all_triples)

    def compare_drugs(self, drug1: str, drug2: str) -> Dict[str, Any]:
        """Compare two drugs across multiple dimensions.

        Args:
            drug1: First drug name
            drug2: Second drug name

        Returns:
            Comparison dictionary
        """
        info1 = self.get_drug_info(drug1)
        info2 = self.get_drug_info(drug2)

        return {
            'drug1': drug1,
            'drug2': drug2,
            'comparison': {
                'indications': {
                    drug1: info1['indications'],
                    drug2: info2['indications'],
                    'overlap': list(set(info1['indications']) & set(info2['indications']))
                },
                'contraindications': {
                    drug1: info1['contraindications'],
                    drug2: info2['contraindications'],
                    'overlap': list(set(info1['contraindications']) & set(info2['contraindications']))
                },
                'side_effects': {
                    drug1: info1['side_effects'],
                    drug2: info2['side_effects'],
                    'overlap': list(set(info1['side_effects']) & set(info2['side_effects']))
                }
            }
        }

    def close(self):
        """Close database connection."""
        self.client.disconnect()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
