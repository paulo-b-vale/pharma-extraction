"""Knowledge triple storage and querying."""

import logging
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
from bson import ObjectId
from pharma_extraction.database.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class KnowledgeStore:
    """Store and query pharmaceutical knowledge triples in MongoDB.

    Knowledge triples are in the format: [entity, relation, value]
    Example: ["Paracetamol", "has_dosage", "500mg"]
    """

    def __init__(self, mongodb_client: MongoDBClient):
        """Initialize knowledge store.

        Args:
            mongodb_client: Connected MongoDB client instance
        """
        self.client = mongodb_client
        self.triples = mongodb_client.get_collection('knowledge_triples')
        self.documents = mongodb_client.get_collection('documents')

    def store_knowledge_graph(
        self,
        document_id: str,
        extraction_results: Dict[str, Any],
        statistics: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store knowledge triples extracted from a document.

        Args:
            document_id: ID of the source document
            extraction_results: Results from knowledge extractor
            statistics: Extraction statistics

        Returns:
            Number of triples stored

        Example:
            >>> store = KnowledgeStore(mongodb_client)
            >>> count = store.store_knowledge_graph(
            ...     doc_id,
            ...     extractor.results['extraction_results']
            ... )
        """
        try:
            triples_to_insert = []

            for phrase_id, result in extraction_results.items():
                if result.get('status') != 'success':
                    continue

                triples = result.get('triples', [])
                phrase_content = result.get('phrase_content', '')
                context = result.get('context', {})

                for triple in triples:
                    # Handle both list and dict formats
                    if isinstance(triple, list) and len(triple) == 3:
                        entity, relation, value = triple
                    elif isinstance(triple, dict):
                        entity = triple.get('entity')
                        relation = triple.get('relation')
                        value = triple.get('value')
                    else:
                        continue

                    # Skip invalid triples
                    if not all([entity, relation, value]):
                        continue

                    triple_doc = {
                        'document_id': document_id,
                        'phrase_id': phrase_id,
                        'entity': entity,
                        'relation': relation,
                        'value': value,
                        'source_phrase': phrase_content,
                        'context': {
                            'section_title': context.get('section_title'),
                            'section_number': context.get('section_number'),
                            'breadcrumb': context.get('breadcrumb'),
                            'page_number': context.get('page_number')
                        },
                        'extracted_date': datetime.utcnow(),
                        'statistics': statistics or {}
                    }

                    triples_to_insert.append(triple_doc)

            if triples_to_insert:
                self.triples.insert_many(triples_to_insert)
                logger.info(f"Stored {len(triples_to_insert)} triples for document {document_id}")
                return len(triples_to_insert)

            return 0

        except Exception as e:
            logger.error(f"Failed to store knowledge graph for document {document_id}: {e}")
            return 0

    def get_triples_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all triples for a document.

        Args:
            document_id: Document ID

        Returns:
            List of knowledge triples
        """
        try:
            triples = list(self.triples.find({'document_id': document_id}))

            # Convert ObjectIds to strings
            for triple in triples:
                triple['_id'] = str(triple['_id'])

            return triples

        except Exception as e:
            logger.error(f"Failed to get triples for document {document_id}: {e}")
            return []

    def search_by_entity(
        self,
        entity: str,
        exact_match: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search triples by entity name.

        Args:
            entity: Entity name to search
            exact_match: Use exact matching (default: text search)
            limit: Maximum number of results

        Returns:
            List of matching triples

        Example:
            >>> triples = store.search_by_entity("Paracetamol")
        """
        try:
            if exact_match:
                query = {'entity': entity}
            else:
                query = {'entity': {'$regex': entity, '$options': 'i'}}

            triples = list(self.triples.find(query).limit(limit))

            for triple in triples:
                triple['_id'] = str(triple['_id'])

            return triples

        except Exception as e:
            logger.error(f"Failed to search by entity '{entity}': {e}")
            return []

    def search_by_relation(
        self,
        relation: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search triples by relation type.

        Args:
            relation: Relation name (e.g., "has_dosage", "is_indicated_for")
            limit: Maximum number of results

        Returns:
            List of matching triples
        """
        try:
            triples = list(
                self.triples.find({'relation': relation}).limit(limit)
            )

            for triple in triples:
                triple['_id'] = str(triple['_id'])

            return triples

        except Exception as e:
            logger.error(f"Failed to search by relation '{relation}': {e}")
            return []

    def find_relationships(
        self,
        entity: str,
        relation: Optional[str] = None
    ) -> List[Tuple[str, str, str]]:
        """Find all relationships for an entity.

        Args:
            entity: Entity name
            relation: Optional relation filter

        Returns:
            List of (entity, relation, value) tuples

        Example:
            >>> rels = store.find_relationships("Paracetamol")
            >>> # Returns: [("Paracetamol", "has_dosage", "500mg"), ...]
        """
        try:
            query = {'entity': {'$regex': entity, '$options': 'i'}}

            if relation:
                query['relation'] = relation

            triples = self.triples.find(query)

            relationships = [
                (t['entity'], t['relation'], t['value'])
                for t in triples
            ]

            return relationships

        except Exception as e:
            logger.error(f"Failed to find relationships for '{entity}': {e}")
            return []

    def get_unique_entities(self) -> List[str]:
        """Get list of all unique entities in the knowledge graph.

        Returns:
            List of unique entity names
        """
        try:
            return self.triples.distinct('entity')
        except Exception as e:
            logger.error(f"Failed to get unique entities: {e}")
            return []

    def get_unique_relations(self) -> List[str]:
        """Get list of all unique relation types.

        Returns:
            List of unique relation names
        """
        try:
            return self.triples.distinct('relation')
        except Exception as e:
            logger.error(f"Failed to get unique relations: {e}")
            return []

    def get_entity_stats(self, entity: str) -> Dict[str, Any]:
        """Get statistics for a specific entity.

        Args:
            entity: Entity name

        Returns:
            Dictionary with entity statistics
        """
        try:
            total_triples = self.triples.count_documents({
                'entity': {'$regex': entity, '$options': 'i'}
            })

            relations = list(self.triples.aggregate([
                {'$match': {'entity': {'$regex': entity, '$options': 'i'}}},
                {'$group': {
                    '_id': '$relation',
                    'count': {'$sum': 1}
                }}
            ]))

            documents = self.triples.distinct(
                'document_id',
                {'entity': {'$regex': entity, '$options': 'i'}}
            )

            return {
                'entity': entity,
                'total_triples': total_triples,
                'relations': {r['_id']: r['count'] for r in relations},
                'document_count': len(documents)
            }

        except Exception as e:
            logger.error(f"Failed to get stats for entity '{entity}': {e}")
            return {}

    def search_knowledge(
        self,
        query: str,
        search_in: str = 'all',
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Full-text search across knowledge triples.

        Args:
            query: Search query
            search_in: Where to search ('entity', 'value', 'all')
            limit: Maximum number of results

        Returns:
            List of matching triples
        """
        try:
            if search_in == 'entity':
                filter_query = {'entity': {'$regex': query, '$options': 'i'}}
            elif search_in == 'value':
                filter_query = {'value': {'$regex': query, '$options': 'i'}}
            else:
                filter_query = {
                    '$or': [
                        {'entity': {'$regex': query, '$options': 'i'}},
                        {'value': {'$regex': query, '$options': 'i'}},
                        {'source_phrase': {'$regex': query, '$options': 'i'}}
                    ]
                }

            triples = list(self.triples.find(filter_query).limit(limit))

            for triple in triples:
                triple['_id'] = str(triple['_id'])

            return triples

        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            return []

    def delete_triples_by_document(self, document_id: str) -> int:
        """Delete all triples for a document.

        Args:
            document_id: Document ID

        Returns:
            Number of triples deleted
        """
        try:
            result = self.triples.delete_many({'document_id': document_id})
            logger.info(f"Deleted {result.deleted_count} triples for document {document_id}")
            return result.deleted_count

        except Exception as e:
            logger.error(f"Failed to delete triples for document {document_id}: {e}")
            return 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            total_triples = self.triples.count_documents({})
            unique_entities = len(self.get_unique_entities())
            unique_relations = len(self.get_unique_relations())

            # Top relations
            top_relations = list(self.triples.aggregate([
                {'$group': {
                    '_id': '$relation',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}},
                {'$limit': 10}
            ]))

            # Top entities
            top_entities = list(self.triples.aggregate([
                {'$group': {
                    '_id': '$entity',
                    'count': {'$sum': 1}
                }},
                {'$sort': {'count': -1}},
                {'$limit': 10}
            ]))

            return {
                'total_triples': total_triples,
                'unique_entities': unique_entities,
                'unique_relations': unique_relations,
                'top_relations': [
                    {'relation': r['_id'], 'count': r['count']}
                    for r in top_relations
                ],
                'top_entities': [
                    {'entity': e['_id'], 'count': e['count']}
                    for e in top_entities
                ]
            }

        except Exception as e:
            logger.error(f"Failed to get knowledge graph statistics: {e}")
            return {}
