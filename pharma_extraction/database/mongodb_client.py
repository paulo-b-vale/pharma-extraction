"""MongoDB client for pharmaceutical data storage."""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING, TEXT
from pymongo.errors import ConnectionFailure, OperationFailure
import os

logger = logging.getLogger(__name__)


class MongoDBClient:
    """MongoDB client for managing pharmaceutical database connections.

    This class handles all MongoDB connections and provides database
    and collection management.

    Attributes:
        client: PyMongo client instance
        db: Current database instance
        db_name: Name of the current database
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: str = "pharma_extraction"
    ):
        """Initialize MongoDB client.

        Args:
            connection_string: MongoDB connection URI (uses env var if None)
            database_name: Database name to use

        Example:
            >>> client = MongoDBClient()
            >>> # Or with custom connection
            >>> client = MongoDBClient("mongodb://localhost:27017")
        """
        self.connection_string = connection_string or os.getenv(
            'MONGODB_URI',
            'mongodb://localhost:27017'
        )
        self.db_name = database_name
        self.client: Optional[MongoClient] = None
        self.db = None

    def connect(self) -> bool:
        """Connect to MongoDB server.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=5000
            )
            # Verify connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB database: {self.db_name}")
            return True

        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            return False

    def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    def is_connected(self) -> bool:
        """Check if connected to MongoDB.

        Returns:
            True if connected, False otherwise
        """
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
        except:
            pass
        return False

    def get_collection(self, collection_name: str):
        """Get a collection from the database.

        Args:
            collection_name: Name of the collection

        Returns:
            MongoDB collection object
        """
        if not self.db:
            raise RuntimeError("Not connected to database. Call connect() first.")
        return self.db[collection_name]

    def create_indexes(self):
        """Create indexes for all collections to optimize queries."""
        try:
            # Documents collection indexes
            docs = self.get_collection('documents')
            docs.create_index([('filename', ASCENDING)], unique=True)
            docs.create_index([('upload_date', DESCENDING)])
            docs.create_index([('metadata.drug_name', TEXT)])
            docs.create_index([('metadata.total_phrases', ASCENDING)])

            # Knowledge triples collection indexes
            triples = self.get_collection('knowledge_triples')
            triples.create_index([('document_id', ASCENDING)])
            triples.create_index([('entity', TEXT)])
            triples.create_index([('relation', ASCENDING)])
            triples.create_index([('value', TEXT)])
            triples.create_index([
                ('entity', TEXT),
                ('relation', TEXT),
                ('value', TEXT)
            ], name='triple_search')

            # Phrases collection indexes
            phrases = self.get_collection('phrases')
            phrases.create_index([('document_id', ASCENDING)])
            phrases.create_index([('phrase_type', ASCENDING)])
            phrases.create_index([('content', TEXT)])
            phrases.create_index([('section_title', ASCENDING)])

            logger.info("Created indexes successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        try:
            stats = self.db.command('dbStats')
            collections_stats = {}

            for collection_name in self.db.list_collection_names():
                col_stats = self.db.command('collStats', collection_name)
                collections_stats[collection_name] = {
                    'count': col_stats.get('count', 0),
                    'size': col_stats.get('size', 0),
                    'avg_obj_size': col_stats.get('avgObjSize', 0)
                }

            return {
                'database_name': self.db_name,
                'collections': collections_stats,
                'total_size': stats.get('dataSize', 0),
                'indexes': stats.get('indexes', 0)
            }

        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}

    def drop_collection(self, collection_name: str) -> bool:
        """Drop a collection (use with caution).

        Args:
            collection_name: Name of collection to drop

        Returns:
            True if successful, False otherwise
        """
        try:
            self.db.drop_collection(collection_name)
            logger.warning(f"Dropped collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop collection {collection_name}: {e}")
            return False

    def reset_database(self) -> bool:
        """Drop all collections (use with extreme caution).

        Returns:
            True if successful, False otherwise
        """
        try:
            for collection_name in self.db.list_collection_names():
                self.db.drop_collection(collection_name)
            logger.warning(f"Reset database: {self.db_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset database: {e}")
            return False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
