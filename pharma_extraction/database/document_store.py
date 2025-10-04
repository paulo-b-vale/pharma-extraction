"""Document storage for parsed pharmaceutical PDFs."""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from bson import ObjectId
from pharma_extraction.database.mongodb_client import MongoDBClient

logger = logging.getLogger(__name__)


class DocumentStore:
    """Store and retrieve parsed pharmaceutical documents in MongoDB.

    This class handles storage of parsed PDF documents with their
    complete structure (phrases, sections, tables, etc.).
    """

    def __init__(self, mongodb_client: MongoDBClient):
        """Initialize document store.

        Args:
            mongodb_client: Connected MongoDB client instance
        """
        self.client = mongodb_client
        self.documents = mongodb_client.get_collection('documents')
        self.phrases = mongodb_client.get_collection('phrases')

    def store_document(
        self,
        filename: str,
        parsed_data: Dict[str, Any],
        pdf_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Store a parsed pharmaceutical document.

        Args:
            filename: Original PDF filename
            parsed_data: Parsed document structure (from parser)
            pdf_path: Path to original PDF file
            metadata: Additional metadata

        Returns:
            Document ID (ObjectId as string) or None if failed

        Example:
            >>> store = DocumentStore(mongodb_client)
            >>> doc_id = store.store_document(
            ...     "paracetamol.pdf",
            ...     parsed_json_data
            ... )
        """
        try:
            # Extract metadata from parsed data
            doc_metadata = parsed_data.get('metadata', {})

            # Merge with additional metadata
            if metadata:
                doc_metadata.update(metadata)

            # Create document record
            document = {
                'filename': filename,
                'pdf_path': pdf_path,
                'upload_date': datetime.utcnow(),
                'metadata': doc_metadata,
                'document_structure': parsed_data.get('document_structure', {}),
                'representations': parsed_data.get('representations', {}),
                'extraction_type': doc_metadata.get('extraction_type', 'unknown')
            }

            # Insert document
            result = self.documents.insert_one(document)
            doc_id = str(result.inserted_id)

            logger.info(f"Stored document: {filename} with ID: {doc_id}")

            # Store phrases separately for better querying
            self._store_phrases(doc_id, parsed_data)

            return doc_id

        except Exception as e:
            logger.error(f"Failed to store document {filename}: {e}")
            return None

    def _store_phrases(self, document_id: str, parsed_data: Dict[str, Any]):
        """Store individual phrases linked to document.

        Args:
            document_id: Parent document ID
            parsed_data: Parsed document data
        """
        try:
            phrase_blocks = (
                parsed_data
                .get('document_structure', {})
                .get('phrase_blocks', [])
            )

            if not phrase_blocks:
                # Try flat_blocks for non-phrase-based parsers
                phrase_blocks = (
                    parsed_data
                    .get('representations', {})
                    .get('flat_blocks', [])
                )

            if phrase_blocks:
                phrases_to_insert = []

                for phrase in phrase_blocks:
                    phrase_doc = {
                        'document_id': document_id,
                        'phrase_id': phrase.get('id'),
                        'content': phrase.get('content'),
                        'phrase_type': phrase.get('metadata', {}).get('phrase_type', 'general'),
                        'section_number': phrase.get('context', {}).get('section_number'),
                        'section_title': phrase.get('context', {}).get('section_title'),
                        'page_number': phrase.get('context', {}).get('page_number'),
                        'breadcrumb': phrase.get('context', {}).get('breadcrumb')
                    }
                    phrases_to_insert.append(phrase_doc)

                if phrases_to_insert:
                    self.phrases.insert_many(phrases_to_insert)
                    logger.info(f"Stored {len(phrases_to_insert)} phrases for document {document_id}")

        except Exception as e:
            logger.error(f"Failed to store phrases for document {document_id}: {e}")

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID.

        Args:
            document_id: Document ID (ObjectId as string)

        Returns:
            Document dictionary or None if not found
        """
        try:
            doc = self.documents.find_one({'_id': ObjectId(document_id)})
            if doc:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
            return doc
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    def get_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by filename.

        Args:
            filename: PDF filename

        Returns:
            Document dictionary or None if not found
        """
        try:
            doc = self.documents.find_one({'filename': filename})
            if doc:
                doc['_id'] = str(doc['_id'])
            return doc
        except Exception as e:
            logger.error(f"Failed to get document by filename {filename}: {e}")
            return None

    def search_documents(
        self,
        query: Optional[str] = None,
        extraction_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search documents.

        Args:
            query: Text search query (searches in drug_name metadata)
            extraction_type: Filter by extraction type (phrase_based, etc.)
            limit: Maximum number of results

        Returns:
            List of matching documents
        """
        try:
            filter_query = {}

            if query:
                filter_query['$text'] = {'$search': query}

            if extraction_type:
                filter_query['extraction_type'] = extraction_type

            docs = list(
                self.documents
                .find(filter_query)
                .limit(limit)
                .sort('upload_date', -1)
            )

            # Convert ObjectIds to strings
            for doc in docs:
                doc['_id'] = str(doc['_id'])

            return docs

        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return []

    def get_phrases_by_document(
        self,
        document_id: str,
        phrase_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all phrases for a document.

        Args:
            document_id: Document ID
            phrase_type: Filter by phrase type (indication, dosage, etc.)

        Returns:
            List of phrases
        """
        try:
            filter_query = {'document_id': document_id}

            if phrase_type:
                filter_query['phrase_type'] = phrase_type

            phrases = list(self.phrases.find(filter_query))

            # Convert ObjectIds to strings
            for phrase in phrases:
                phrase['_id'] = str(phrase['_id'])

            return phrases

        except Exception as e:
            logger.error(f"Failed to get phrases for document {document_id}: {e}")
            return []

    def search_phrases(
        self,
        query: str,
        phrase_type: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search phrases by content.

        Args:
            query: Text search query
            phrase_type: Filter by phrase type
            limit: Maximum number of results

        Returns:
            List of matching phrases
        """
        try:
            filter_query = {'$text': {'$search': query}}

            if phrase_type:
                filter_query['phrase_type'] = phrase_type

            phrases = list(
                self.phrases
                .find(filter_query)
                .limit(limit)
            )

            # Convert ObjectIds to strings
            for phrase in phrases:
                phrase['_id'] = str(phrase['_id'])

            return phrases

        except Exception as e:
            logger.error(f"Failed to search phrases: {e}")
            return []

    def update_document_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update document metadata.

        Args:
            document_id: Document ID
            metadata: Metadata fields to update

        Returns:
            True if successful, False otherwise
        """
        try:
            result = self.documents.update_one(
                {'_id': ObjectId(document_id)},
                {'$set': {'metadata': metadata}}
            )
            return result.modified_count > 0

        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its phrases.

        Args:
            document_id: Document ID

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete phrases first
            self.phrases.delete_many({'document_id': document_id})

            # Delete document
            result = self.documents.delete_one({'_id': ObjectId(document_id)})

            logger.info(f"Deleted document {document_id}")
            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics.

        Returns:
            Dictionary with statistics
        """
        try:
            return {
                'total_documents': self.documents.count_documents({}),
                'total_phrases': self.phrases.count_documents({}),
                'extraction_types': list(
                    self.documents.distinct('extraction_type')
                ),
                'latest_upload': self.documents.find_one(
                    {},
                    sort=[('upload_date', -1)]
                )
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
