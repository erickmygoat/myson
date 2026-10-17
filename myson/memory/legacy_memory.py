"""
This module provides Long-Term Memory capabilities using a Vector Database (ChromaDB).
"""
import logging
import time
import os

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

class Memory:
    """
    The agent's long-term memory system.
    Stores observations, conversations, and facts.
    """

    def __init__(self, persist_directory="./brain_memory"):
        self.logger = logging.getLogger(__name__)
        self.client = None
        self.collection = None

        if CHROMA_AVAILABLE:
            try:
                self.client = chromadb.PersistentClient(path=persist_directory)
                self.collection = self.client.get_or_create_collection(name="episodic_memory")
                self.rules_collection = self.client.get_or_create_collection(name="learned_rules")
                print("Memory: Long-term memory initialized.")
            except Exception as e:
                self.logger.error(f"Failed to init ChromaDB: {e}")
                self.client = None
        else:
            self.logger.warning("ChromaDB not installed. Memory will be transient/disabled.")

    def store_rule(self, rule_text, topic):
        """
        Stores a learned rule into the semantic memory.
        """
        if not CHROMA_AVAILABLE or not self.client:
            return

        try:
            rule_id = f"rule_{int(time.time() * 1000)}_{hash(rule_text)}"
            self.rules_collection.add(
                documents=[rule_text],
                metadatas=[{"topic": topic, "timestamp": time.time()}],
                ids=[rule_id]
            )
        except Exception as e:
            self.logger.error(f"Failed to store rule: {e}")

    def retrieve_relevant_rules(self, query, n_results=5):
        """
        Retrieves rules relevant to the current context.
        """
        if not CHROMA_AVAILABLE or not self.client:
            return []

        try:
            results = self.rules_collection.query(
                query_texts=[query],
                n_results=n_results
            )
            if results and results['documents']:
                return results['documents'][0]
            return []
        except Exception as e:
            self.logger.error(f"Failed to retrieve rules: {e}")
            return []

    def reset_rules(self):
        """
        Clears the rules collection. Used during consolidation.
        """
        if not CHROMA_AVAILABLE or not self.client:
            return
        try:
            self.client.delete_collection("learned_rules")
            self.rules_collection = self.client.get_or_create_collection(name="learned_rules")
        except Exception as e:
            self.logger.error(f"Failed to reset rules: {e}")

    def save_context(self, input_text, output_text):
        """
        Stores an interaction pair into memory.
        """
        if not input_text and not output_text:
            return

        text_to_store = ""
        if input_text:
             text_to_store += f"User: {input_text}\n"
        if output_text:
             text_to_store += f"Agent: {output_text}\n"

        self.store_memory(text_to_store.strip())

    def store_memory(self, text, metadata=None):
        """
        Stores a text snippet into memory.
        """
        if not CHROMA_AVAILABLE or not self.collection or not text:
            return

        if metadata is None:
            metadata = {}

        metadata["timestamp"] = time.time()

        try:
            # Simple unique ID generation based on timestamp + hash to avoid collisions in bulk
            mem_id = f"mem_{int(time.time() * 1000)}_{hash(text)}"
            self.collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[mem_id]
            )
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")

    def retrieve_context(self, query, n_results=3):
        """
        Retrieves relevant memories for a given query.
        """
        if not CHROMA_AVAILABLE or not self.collection or not query:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            if results and results['documents']:
                return results['documents'][0]
            return []
        except Exception as e:
            self.logger.error(f"Failed to retrieve memory: {e}")
            return []

    def ingest_bulk_folder(self, folder_path):
        """
        Scans a folder and adds all supported documents to memory.
        """
        if not os.path.exists(folder_path):
            print(f"Memory: Folder {folder_path} does not exist.")
            return

        print(f"Memory: Ingesting folder {folder_path}...")
        count = 0

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith((".txt", ".py", ".md", ".json")):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()

                        if content.strip():
                            # Chunking could be added here for large files
                            self.store_memory(content, metadata={"source": file_path})
                            count += 1
                    except Exception as e:
                        print(f"Memory: Failed to ingest {file}: {e}")

        print(f"Memory: Bulk ingestion complete. {count} documents added.")
