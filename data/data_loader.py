import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset, get_dataset_config_names
from utils.logger import logger
from utils.text_processing import TextProcessor
from config import config


class LegalDataLoader:
    """Handles loading and preprocessing of authoritative legal datasets."""

    def __init__(self):
        self.text_processor = TextProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.cache_dir = Path("./data/processed")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "processed_legal_docs.json"

        # Get valid configurations from environment or use defaults
        self.valid_configs = self._get_valid_configs_from_env()

    def _get_valid_configs_from_env(self) -> Dict[str, List[str]]:
        """Get valid configurations from environment variables or use defaults."""
        # Get multiple configs from environment
        multiple_configs_str = getattr(config, 'multiple_configs', None)

        if multiple_configs_str:
            # Parse comma-separated configs from environment
            env_configs = [name.strip() for name in multiple_configs_str.split(',')]
            dataset_name = getattr(config, 'legal_dataset', 'pile-of-law/pile-of-law')

            return {
                dataset_name: env_configs
            }

        # Fallback to defaults if no environment configs
        return {
            "pile-of-law/pile-of-law": [
                "federal_register",
                "cfr",
                "uscode",
                "congressional_hearings",
                "olc_memos",
                "fre",
                "frcp",
                "uniform_commercial_code"
            ],
            "lexlms/lex_files": [
                "statutes",
                "regulations",
                "case_law"
            ]
        }

    def get_available_configs(self, dataset_name: str) -> List[str]:
        """Get available configurations for a dataset."""
        try:
            if dataset_name in self.valid_configs:
                return self.valid_configs[dataset_name]

            # Try to get configs from HuggingFace
            configs = get_dataset_config_names(dataset_name)
            logger.info(f"Available configs for {dataset_name}: {configs}")
            return configs
        except Exception as e:
            logger.warning(f"Could not get configs for {dataset_name}: {e}")
            return []

    def validate_configs(self, dataset_name: str, config_names: List[str]) -> List[str]:
        """Validate and filter configuration names."""
        available_configs = self.get_available_configs(dataset_name)

        if not available_configs:
            logger.warning(f"No available configs found for {dataset_name}")
            return []

        valid_configs = []
        for config_name in config_names:
            if config_name in available_configs:
                valid_configs.append(config_name)
            else:
                logger.warning(f"Config '{config_name}' not found in {dataset_name}. Available: {available_configs}")

        if not valid_configs:
            logger.info(f"No valid configs specified, using first available: {available_configs[0]}")
            valid_configs = [available_configs[0]]

        return valid_configs

    def load_authoritative_legal_data(self, dataset_name: str = None, config_name: str = None, max_docs: int = None) -> \
    List[Dict[str, Any]]:
        """Load authoritative legal documents from HuggingFace datasets."""
        dataset_name = dataset_name or config.legal_dataset
        config_name = config_name or getattr(config, 'legal_dataset_config', None)
        max_docs = max_docs or config.max_documents

        logger.info(f"Loading authoritative legal dataset: {dataset_name}")
        logger.info(f"Dataset config: {config_name}")
        logger.info(f"Max documents to load: {max_docs}")

        if not dataset_name:
            logger.error("No dataset name provided in config.legal_dataset")
            return self._load_fallback_data()

        # Validate config before attempting to load
        if config_name:
            valid_configs = self.validate_configs(dataset_name, [config_name])
            if not valid_configs:
                logger.error(f"Invalid config '{config_name}' for dataset '{dataset_name}'")
                return self._load_fallback_data()
            config_name = valid_configs[0]

        try:
            # Handle different dataset structures
            if dataset_name == "pile-of-law/pile-of-law":
                return self._load_pile_of_law(config_name, max_docs)
            elif dataset_name == "lexlms/lex_files":
                return self._load_lex_files(config_name, max_docs)
            elif dataset_name == "pile-of-law/us-court-opinions":
                return self._load_court_opinions(max_docs)
            elif dataset_name == "jonathanli/lawbench":
                return self._load_lawbench(config_name, max_docs)
            else:
                return self._load_generic_legal_dataset(dataset_name, config_name, max_docs)

        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._load_fallback_data()

    def _load_pile_of_law(self, config_name: str = None, max_docs: int = None) -> List[Dict[str, Any]]:
        """Load from Pile of Law dataset - contains actual legal texts."""
        try:
            # Use default config if none provided
            subset = config_name or "federal_register"  # Default to federal register

            # Validate the subset exists
            valid_configs = self.validate_configs("pile-of-law/pile-of-law", [subset])
            if not valid_configs:
                logger.error(f"Invalid Pile of Law config: {subset}")
                return []
            subset = valid_configs[0]

            logger.info(f"Loading Pile of Law subset: {subset}")

            # Load dataset with appropriate split
            if max_docs and max_docs > 0:
                split_str = f"train[:{max_docs}]"
            else:
                split_str = "train"

            # Remove trust_remote_code parameter as it's not supported by this dataset
            dataset = load_dataset(
                "pile-of-law/pile-of-law",
                subset,
                split=split_str,
                streaming=False
            )

            legal_documents = []
            for i, item in enumerate(dataset):
                try:
                    text_content = item.get('text', '')
                    if len(text_content) < 200:  # Skip very short documents
                        continue

                    doc = {
                        "id": f"pol_{subset}_{i}",
                        "text": text_content,
                        "title": self._extract_title_from_text(text_content) or f"Legal Document {i}",
                        "source": f"pile-of-law/{subset}",
                        "jurisdiction": "US",
                        "type": "authoritative_legal_text",
                        "authority_level": "federal" if "federal" in subset else "general",
                        "document_category": subset
                    }

                    legal_documents.append(doc)

                except Exception as e:
                    logger.warning(f"Error processing Pile of Law item {i}: {e}")
                    continue

            logger.info(f"Loaded {len(legal_documents)} documents from Pile of Law")
            return legal_documents

        except Exception as e:
            logger.error(f"Failed to load Pile of Law: {e}")
            return []

    def _load_lex_files(self, config_name: str = None, max_docs: int = None) -> List[Dict[str, Any]]:
        """Load from LexFiles dataset - legal case files and statutes."""
        try:
            subset = config_name or "statutes"

            # Validate the subset exists
            valid_configs = self.validate_configs("lexlms/lex_files", [subset])
            if not valid_configs:
                logger.error(f"Invalid LexFiles config: {subset}")
                return []
            subset = valid_configs[0]

            logger.info(f"Loading LexFiles subset: {subset}")

            if max_docs and max_docs > 0:
                split_str = f"train[:{max_docs}]"
            else:
                split_str = "train"

            dataset = load_dataset(
                "lexlms/lex_files",
                subset,
                split=split_str,
                streaming=False
            )

            legal_documents = []
            for i, item in enumerate(dataset):
                try:
                    # LexFiles typically has 'text' and metadata
                    text_content = item.get('text', item.get('content', ''))
                    if len(text_content) < 200:
                        continue

                    doc = {
                        "id": f"lexfiles_{subset}_{i}",
                        "text": text_content,
                        "title": item.get('title', f"Legal Document {i}"),
                        "source": f"lexlms/lex_files/{subset}",
                        "jurisdiction": item.get('jurisdiction', 'US'),
                        "type": "legal_statute" if subset == "statutes" else "case_law",
                        "authority_level": "authoritative",
                        "document_category": subset
                    }

                    legal_documents.append(doc)

                except Exception as e:
                    logger.warning(f"Error processing LexFiles item {i}: {e}")
                    continue

            logger.info(f"Loaded {len(legal_documents)} documents from LexFiles")
            return legal_documents

        except Exception as e:
            logger.error(f"Failed to load LexFiles: {e}")
            return []

    def _load_court_opinions(self, max_docs: int = None) -> List[Dict[str, Any]]:
        """Load US court opinions - actual case law."""
        try:
            logger.info("Loading US Court Opinions")

            if max_docs and max_docs > 0:
                split_str = f"train[:{max_docs}]"
            else:
                split_str = "train"

            dataset = load_dataset(
                "pile-of-law/us-court-opinions",
                split=split_str,
                streaming=False
            )

            legal_documents = []
            for i, item in enumerate(dataset):
                try:
                    text_content = item.get('text', item.get('opinion_text', ''))
                    if len(text_content) < 500:  # Court opinions should be substantial
                        continue

                    doc = {
                        "id": f"court_opinion_{i}",
                        "text": text_content,
                        "title": item.get('case_name', item.get('title', f"Court Opinion {i}")),
                        "source": "pile-of-law/us-court-opinions",
                        "jurisdiction": item.get('court', 'US'),
                        "type": "case_law",
                        "authority_level": "judicial_precedent",
                        "court_level": item.get('court_level', 'unknown'),
                        "date": item.get('date', '')
                    }

                    legal_documents.append(doc)

                except Exception as e:
                    logger.warning(f"Error processing court opinion {i}: {e}")
                    continue

            logger.info(f"Loaded {len(legal_documents)} court opinions")
            return legal_documents

        except Exception as e:
            logger.error(f"Failed to load court opinions: {e}")
            return []

    def _load_lawbench(self, config_name: str = None, max_docs: int = None) -> List[Dict[str, Any]]:
        """Load from LawBench dataset - legal knowledge base."""
        try:
            # LawBench has different tasks, we want knowledge-based ones
            subset = config_name or "statute_law"

            logger.info(f"Loading LawBench subset: {subset}")

            if max_docs and max_docs > 0:
                split_str = f"train[:{max_docs}]"
            else:
                split_str = "train"

            dataset = load_dataset(
                "jonathanli/lawbench",
                subset,
                split=split_str,
                streaming=False
            )

            legal_documents = []
            for i, item in enumerate(dataset):
                try:
                    # Extract authoritative content, not questions
                    text_content = item.get('context', item.get('passage', item.get('text', '')))

                    # Skip if it looks like a question rather than authoritative text
                    if self._is_question_format(text_content):
                        continue

                    if len(text_content) < 200:
                        continue

                    doc = {
                        "id": f"lawbench_{subset}_{i}",
                        "text": text_content,
                        "title": item.get('title', f"Legal Knowledge {i}"),
                        "source": f"jonathanli/lawbench/{subset}",
                        "jurisdiction": item.get('jurisdiction', 'US'),
                        "type": "legal_knowledge",
                        "authority_level": "educational",
                        "document_category": subset
                    }

                    legal_documents.append(doc)

                except Exception as e:
                    logger.warning(f"Error processing LawBench item {i}: {e}")
                    continue

            logger.info(f"Loaded {len(legal_documents)} documents from LawBench")
            return legal_documents

        except Exception as e:
            logger.error(f"Failed to load LawBench: {e}")
            return []

    def _load_generic_legal_dataset(self, dataset_name: str, config_name: str = None, max_docs: int = None) -> List[
        Dict[str, Any]]:
        """Generic loader for other legal datasets."""
        try:
            logger.info(f"Loading generic legal dataset: {dataset_name}")

            if max_docs and max_docs > 0:
                split_str = f"train[:{max_docs}]"
            else:
                split_str = "train"

            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split_str, streaming=False)
            else:
                dataset = load_dataset(dataset_name, split=split_str, streaming=False)

            legal_documents = []
            for i, item in enumerate(dataset):
                try:
                    text_content = self._extract_authoritative_content(item)

                    if not text_content or len(text_content) < 200:
                        continue

                    # Skip question-format content
                    if self._is_question_format(text_content):
                        continue

                    doc = {
                        "id": f"generic_{i}",
                        "text": text_content,
                        "title": self._extract_title(item, i),
                        "source": f"{dataset_name}/{config_name}" if config_name else dataset_name,
                        "jurisdiction": item.get("jurisdiction", "US"),
                        "type": "legal_document",
                        "authority_level": "general"
                    }

                    legal_documents.append(doc)

                except Exception as e:
                    logger.warning(f"Error processing generic item {i}: {e}")
                    continue

            logger.info(f"Loaded {len(legal_documents)} documents from generic dataset")
            return legal_documents

        except Exception as e:
            logger.error(f"Failed to load generic dataset: {e}")
            return []

    def _is_question_format(self, text: str) -> bool:
        """Check if text appears to be a question or user scenario rather than authoritative legal text."""
        if not text:
            return True

        text_lower = text.lower().strip()

        # Indicators of question/scenario format
        question_indicators = [
            "so i help", "i need help", "my landlord", "my employer",
            "what should i do", "can someone", "i was wondering",
            "my situation", "please help", "advice needed",
            "question:", "my question", "i have a question"
        ]

        # Check for question patterns
        if any(indicator in text_lower for indicator in question_indicators):
            return True

        # Check if it starts with common question words
        question_starters = ["what", "how", "when", "where", "why", "can", "should", "would", "could"]
        first_words = text_lower.split()[:3]
        if any(word in question_starters for word in first_words):
            return True

        # Check for personal pronouns that indicate user scenarios
        personal_pronouns = ["i ", "my ", "me ", "we ", "our "]
        if any(pronoun in text_lower[:100] for pronoun in personal_pronouns):
            return True

        return False

    def _extract_authoritative_content(self, item: Dict) -> str:
        """Extract authoritative legal content, avoiding user questions."""
        # Prioritize fields that typically contain authoritative content
        authoritative_fields = [
            'statute_text', 'regulation_text', 'case_text', 'legal_text',
            'content', 'text', 'body', 'document', 'passage', 'context'
        ]

        for field in authoritative_fields:
            if field in item and item[field]:
                content = str(item[field])
                if not self._is_question_format(content):
                    return content

        return ""

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract title from the beginning of legal text."""
        if not text:
            return None

        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 200:  # Reasonable title length
                # Skip if it looks like a question
                if not self._is_question_format(line):
                    return line

        return None

    def load_multiple_authoritative_configs(self, dataset_name: str = None, config_names: List[str] = None,
                                            max_docs_per_config: int = None) -> List[Dict[str, Any]]:
        """Load documents from multiple authoritative dataset configurations."""
        dataset_name = dataset_name or config.legal_dataset

        # Get config names from environment variable or use defaults
        if not config_names:
            # Check if multiple configs are specified in environment
            multiple_configs_str = getattr(config, 'multiple_configs', None)
            if multiple_configs_str:
                config_names = [name.strip() for name in multiple_configs_str.split(',')]
            else:
                # Default to authoritative legal datasets
                if dataset_name == "pile-of-law/pile-of-law":
                    config_names = [
                        'federal_register',
                        'cfr',
                        'uscode'
                    ]
                elif dataset_name == "lexlms/lex_files":
                    config_names = [
                        'statutes',
                        'regulations',
                        'case_law'
                    ]
                else:
                    # Generic fallback
                    config_names = ['default']

        # Validate all config names before proceeding
        valid_config_names = self.validate_configs(dataset_name, config_names)
        if not valid_config_names:
            logger.error(f"No valid configurations found for {dataset_name}")
            return self._load_fallback_data()

        if max_docs_per_config is None:
            max_docs_per_config = max(1, config.max_documents // len(valid_config_names))

        all_documents = []

        for config_name in valid_config_names:
            try:
                logger.info(f"Loading authoritative config: {config_name}")
                documents = self.load_authoritative_legal_data(
                    dataset_name=dataset_name,
                    config_name=config_name,
                    max_docs=max_docs_per_config
                )

                # Filter out any remaining question-format content
                authoritative_docs = [doc for doc in documents if not self._is_question_format(doc['text'])]

                all_documents.extend(authoritative_docs)
                logger.info(f"Loaded {len(authoritative_docs)} authoritative documents from {config_name}")

            except Exception as e:
                logger.error(f"Failed to load config {config_name}: {e}")
                continue

        logger.info(f"Total authoritative documents loaded: {len(all_documents)}")

        if len(all_documents) == 0:
            logger.warning("No authoritative documents loaded, using fallback data")
            return self._load_fallback_data()

        return all_documents

    def _extract_title(self, item: Dict, index: int) -> str:
        """Extract title from dataset item with flexible field names."""
        title_fields = ['title', 'name', 'heading', 'subject', 'case_name', 'statute_name']

        for field in title_fields:
            if field in item and item[field]:
                title = str(item[field])
                if len(title) > 100:
                    title = title[:97] + "..."
                return title

        return f"Legal Document {index}"

    def _load_fallback_data(self) -> List[Dict[str, Any]]:
        """Load authoritative fallback legal data."""
        logger.info("Loading authoritative fallback legal data")

        fallback_docs = [
            {
                "id": "fallback_flsa",
                "text": """
                Fair Labor Standards Act (29 U.S.C. § 207): Employees covered by the Act must receive overtime pay 
                for hours worked over 40 in a workweek at a rate not less than time and one-half their regular rates 
                of pay. There is no limit on the number of hours employees 16 years or older may work in any workweek. 
                The Act does not require overtime pay for work on weekends, holidays, or regular days of rest, unless 
                overtime is worked on such days. The Act applies only to overtime work. Extra pay for work on weekends 
                or nights is a matter of agreement between the employer and the employee (or the employee's representative). 
                The Act requires that overtime must be paid at the rate of time and one-half the employee's regular rate 
                for all hours worked in excess of 40 hours in a workweek.
                """,
                "title": "Fair Labor Standards Act - Overtime Provisions",
                "source": "29_USC_207",
                "jurisdiction": "US_Federal",
                "type": "federal_statute",
                "authority_level": "federal_law"
            },
            {
                "id": "fallback_housing",
                "text": """
                Under the Federal Fair Housing Act (42 U.S.C. § 3604), it is unlawful to refuse to sell or rent after 
                the making of a bona fide offer, or to refuse to negotiate for the sale or rental of, or otherwise make 
                unavailable or deny, a dwelling to any person because of race, color, religion, sex, familial status, 
                or national origin. It is also unlawful to discriminate in the terms, conditions, or privileges of sale 
                or rental of a dwelling, or in the provision of services or facilities in connection therewith, because 
                of race, color, religion, sex, familial status, or national origin. The Act also prohibits discriminatory 
                advertising and blockbusting practices. State and local fair housing laws may provide additional protections.
                """,
                "title": "Fair Housing Act - Discrimination Prohibitions",
                "source": "42_USC_3604",
                "jurisdiction": "US_Federal",
                "type": "federal_statute",
                "authority_level": "federal_law"
            },
            {
                "id": "fallback_contracts",
                "text": """
                Under general contract law principles recognized across U.S. jurisdictions, a valid contract requires: 
                (1) Offer - a definite proposal made by one party to another indicating a willingness to enter into a 
                contract; (2) Acceptance - an unqualified agreement to the terms of the offer; (3) Consideration - 
                something of legal value given in exchange for the promise; (4) Legal capacity - the parties must have 
                the legal ability to enter into a contract; and (5) Legal purpose - the contract must be for a lawful 
                purpose. These elements must be present for a contract to be legally enforceable. Contracts may be 
                express (stated in words) or implied (inferred from conduct). The Uniform Commercial Code (UCC) 
                governs contracts for the sale of goods in most states.
                """,
                "title": "Contract Formation Elements",
                "source": "Restatement_of_Contracts",
                "jurisdiction": "US_General",
                "type": "legal_principle",
                "authority_level": "common_law"
            }
        ]

        logger.info(f"Loaded {len(fallback_docs)} authoritative fallback documents")
        return fallback_docs

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents into chunks suitable for embedding."""
        cached_data = self._load_from_cache()
        if cached_data:
            logger.info("Loading processed documents from cache")
            return cached_data

        processed_chunks = []

        logger.info(f"Processing {len(documents)} documents...")
        for i, doc in enumerate(documents):
            try:
                chunks = self.text_processor.preprocess_legal_document(doc)
                processed_chunks.extend(chunks)

                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(documents)} documents")

            except Exception as e:
                logger.error(f"Error processing document {doc.get('id', 'unknown')}: {e}")
                continue

        if processed_chunks:
            self._save_to_cache(processed_chunks)

        logger.info(f"Processed {len(documents)} documents into {len(processed_chunks)} chunks")
        return processed_chunks

    def _save_to_cache(self, chunks: List[Dict[str, Any]]):
        """Save processed chunks to cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(chunks)} chunks to cache at {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    def _load_from_cache(self) -> Optional[List[Dict[str, Any]]]:
        """Load processed chunks from cache."""
        if not self.cache_file.exists():
            logger.debug(f"Cache file does not exist: {self.cache_file}")
            return None

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"Loaded {len(chunks)} chunks from cache")
            return chunks
        except Exception as e:
            logger.error(f"Error loading from cache: {e}")
            return None

    def clear_cache(self):
        """Clear the processed documents cache."""
        try:
            if self.cache_file.exists():
                self.cache_file.unlink()
                logger.info("Cache cleared successfully")
            else:
                logger.info("No cache file to clear")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_processed_legal_data(self, force_reload: bool = False, use_multiple_configs: bool = True) -> List[
        Dict[str, Any]]:
        """Main method to get processed authoritative legal data."""
        if force_reload:
            logger.info("Force reload requested - clearing cache")
            self.clear_cache()

        if not force_reload:
            cached_data = self._load_from_cache()
            if cached_data:
                return cached_data

        logger.info("Loading fresh authoritative legal data from source...")

        if use_multiple_configs:
            raw_documents = self.load_multiple_authoritative_configs()
        else:
            raw_documents = self.load_authoritative_legal_data()

        return self.process_documents(raw_documents)

    def validate_config(self) -> bool:
        """Validate that the configuration has the required fields."""
        required_fields = ['legal_dataset', 'max_documents', 'chunk_size', 'chunk_overlap']
        missing_fields = []

        for field in required_fields:
            if not hasattr(config, field) or getattr(config, field) is None:
                missing_fields.append(field)

        if missing_fields:
            logger.error(f"Missing required config fields: {missing_fields}")
            return False

        logger.info(f"Config validation passed. Dataset: {config.legal_dataset}")

        # Validate that we're using an authoritative dataset
        authoritative_datasets = [
            "pile-of-law/pile-of-law",
            "lexlms/lex_files",
            "pile-of-law/us-court-opinions",
            "jonathanli/lawbench"
        ]

        if config.legal_dataset not in authoritative_datasets:
            logger.warning(f"Dataset {config.legal_dataset} may not contain authoritative legal sources")
            logger.warning(f"Recommended datasets: {authoritative_datasets}")

        return True