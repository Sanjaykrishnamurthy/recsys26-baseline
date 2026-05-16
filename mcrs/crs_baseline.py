import os
import torch
from typing import Optional, Any, List, Dict
from mcrs.db_item import MusicCatalogDB
from mcrs.db_user import UserProfileDB
from mcrs.lm_modules import load_lm_module
from mcrs.retrieval_modules import load_retrieval_module

class CRS_BASELINE:
    """
    Conversational Recommender System (CRS) baseline that wires together an LLM module and an item retrieval module over a music catalog and user profiles.
    Attributes:
        cache_dir: Local path for caching artifacts and indices.
        lm_type: Identifier/name for the LLM backend to load.
        retrieval_type: Retrieval backend to use (e.g., "bm25", "bert").
        item_db_name: Hugging Face dataset or DB name for item metadata.
        user_db_name: Hugging Face dataset or DB name for user metadata.
        corpus_types: Item fields used for retrieval (e.g., title, artist, album).
        device: Compute device for the LLM (e.g., "cuda", "cpu").
        dtype: Torch dtype used by the LLM.
        use_user_reranker: Whether to rerank Stage 1 candidates by user-profile
            BERT similarity before passing the top-1 to the LLM. Requires
            retrieval_type="bert".
        lm: Loaded LLM module used for response generation.
        retrieval: Retrieval module used to fetch candidate items.
        reranker: Optional UserBERTReranker for personalized reranking.
        item_db: Item metadata database accessor.
        user_db: User profile database accessor.
        prompts_dir: Directory containing prompt templates.
        role_prompt: Loaded prompt templates keyed by role.
        session_memory: In-memory list of message dicts for the current session.
    """
    def __init__(self,
        lm_type="meta-llama/Llama-3.2-1B-Instruct",
        retrieval_type="bm25",
        item_db_name: str = "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        user_db_name: str = "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
        track_split_types: list[str] = ["all_tracks"],
        user_split_types: list[str] = ["all_users"],
        corpus_types: list[str] = ["track_name", "artist_name", "album_name"],
        cache_dir="./cache",
        device="cuda",
        attn_implementation="eager",
        dtype=torch.bfloat16,
        use_user_reranker: bool = False,
        retrieval_model_name: str | None = None,
    ):
        """Initialize the CRS baseline components.

        Args:
            lm_type: LLM model identifier to load for response generation.
                     Any Hugging Face causal LM is supported (Llama, Qwen, etc.).
            retrieval_type: Retrieval backend name ("bm25", "sbert", or "bert").
            item_db_name: Dataset/DB name for item metadata.
            user_db_name: Dataset/DB name for user metadata.
            track_split_types: Dataset split names to load. Must be ["all_tracks"].
            user_split_types: User dataset split names to load.
            corpus_types: Item metadata fields used for retrieval indexing.
            cache_dir: Local directory for caching artifacts/indices.
            device: Compute device for the LLM (e.g., "cuda", "mps", "cpu").
            attn_implementation: Attention backend ("eager" or "flash_attention_2").
            dtype: Torch dtype for the LLM weights/tensors.
            use_user_reranker: If True, attaches a UserBERTReranker that reranks
                the Stage 1 candidate pool by cosine similarity between the user
                profile embedding and each candidate track embedding. Only
                effective when retrieval_type="bert"; a warning is printed and
                the flag is ignored for BM25.
            retrieval_model_name: Optional model name or local path for the
                retrieval encoder (SBERT/BERT). Defaults to the module's built-in
                default when None. Use a local path for Kaggle notebooks.
        """
        self.cache_dir = cache_dir
        self.lm_type = lm_type
        self.retrieval_type = retrieval_type
        self.item_db_name = item_db_name
        self.user_db_name = user_db_name
        self.track_split_types = track_split_types
        self.user_split_types = user_split_types
        self.corpus_types = corpus_types
        self.device = device
        self.dtype = dtype
        self.attn_implementation = attn_implementation
        self.use_user_reranker = use_user_reranker
        self.retrieval_model_name = retrieval_model_name

        self.lm = load_lm_module(self.lm_type, self.device, self.attn_implementation, self.dtype)
        self.retrieval = load_retrieval_module(self.retrieval_type, self.item_db_name, self.track_split_types, self.corpus_types, self.cache_dir, retrieval_model_name=self.retrieval_model_name)
        self.item_db = MusicCatalogDB(self.item_db_name, self.track_split_types, self.corpus_types)
        self.user_db = UserProfileDB(self.user_db_name, self.user_split_types)
        self.prompts_dir = os.path.join(os.path.dirname(__file__), "system_prompts")
        self.role_prompt = {
            "role_play": open(f"{self.prompts_dir}/roleplay.txt", "r", encoding="utf-8").read(),
            "personalization": open(f"{self.prompts_dir}/personalization.txt", "r", encoding="utf-8").read(),
            "response_generation": open(f"{self.prompts_dir}/response_generation.txt", "r", encoding="utf-8").read(),
        }
        self.session_memory = []

        # Optional user-profile reranker — only works with BERT retrieval
        self.reranker = None
        if use_user_reranker:
            if retrieval_type != "bert":
                print(
                    "[WARNING] use_user_reranker=True requires retrieval_type='bert'. "
                    f"Got '{retrieval_type}' — reranker will be disabled."
                )
            else:
                from mcrs.retrieval_modules import UserBERTReranker
                self.reranker = UserBERTReranker(self.retrieval)
                print("[INFO] UserBERTReranker enabled — candidates will be reranked by user profile similarity.")

    def _reset_session_memory(self):
        """Clear all messages stored in the current session memory.
        """
        self.session_memory = []

    def _upload_session_memory(self, chat_history: List[Dict[str, Any]]):
        """Upload the session memory to the database.
        """
        self.session_memory = chat_history

    def _get_system_prompt(self, user_id: Optional[str] = None) -> str:
        """Build the system prompt, optionally personalized with a user profile.
        Args:
            user_id: Optional user identifier. When provided, includes a personalization segment derived from the user's profile.
        Returns:
            The final system prompt string used for the LLM.
        """
        system_prompt = self.role_prompt["role_play"] + self.role_prompt["response_generation"]
        if user_id:
            user_profile_str = self.user_db.id_to_profile_str(user_id)
            system_prompt += self.role_prompt["personalization"] + '\n' + user_profile_str
        return system_prompt

    def _get_user_profile_str(self, user_id: Optional[str]) -> Optional[str]:
        """Return the user profile string for reranking, or None if unavailable.
        Args:
            user_id: Optional user identifier.
        Returns:
            Plain-text user profile string, or None.
        """
        if user_id:
            return self.user_db.id_to_profile_str(user_id)
        return None

    def chat(self, user_query: str, user_id: Optional[str] = None) -> dict[str, Any]:
        """Run a single CRS turn: retrieve items and generate a response.
        Args:
            user_query: The user's latest message or request.
            user_id: Optional user identifier for personalization.
        Returns:
            A dictionary with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        self.session_memory.append({"role": "user", "content": user_query})
        # stage0. system prompt
        system_prompt = self._get_system_prompt(user_id)
        # stage1. retrieval — use latest user query only to avoid truncation
        #         and embedding dilution with dense retrievers (SBERT/BERT)
        retrieval_input = user_query
        retrieval_topk = 100 if self.reranker else 20
        retrieval_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=retrieval_topk)
        # stage1b. optional user-profile reranking
        if self.reranker:
            user_profile = self._get_user_profile_str(user_id)
            if user_profile:
                retrieval_items = self.reranker.rerank(user_profile, retrieval_items, topk=20)
            else:
                retrieval_items = retrieval_items[:20]
        recommend_item = self.item_db.id_to_metadata(retrieval_items[0])
        # stage2. response generation
        response = self.lm.response_generation(system_prompt, self.session_memory, recommend_item)
        return {
            "user_id": user_id,
            "user_query": user_query,
            "retrieval_items": retrieval_items,
            "recommend_item": recommend_item,
            "response": response,
        }

    def batch_chat(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run multiple CRS turns in batch: retrieve items and generate responses.
        Args:
            batch_data: List of dictionaries, each containing:
                - user_query: The user's latest message or request.
                - user_id: Optional user identifier for personalization.
                - session_memory: List of chat history messages.
        Returns:
            A list of dictionaries, each with keys:
                - user_id: The user identifier (may be None).
                - user_query: Echo of the input query.
                - retrieval_items: List of retrieved item IDs (top candidates).
                - recommend_item: Metadata for the top recommended item.
                - response: The generated assistant response string.
        """
        # Prepare batch inputs
        sys_prompts = []
        retrieval_inputs = []
        session_memories = []
        user_ids = []

        for data in batch_data:
            user_query = data['user_query']
            user_id = data.get('user_id')
            session_memory = data['session_memory'].copy()
            session_memory.append({"role": "user", "content": user_query})

            sys_prompts.append(self._get_system_prompt(user_id))
            # Use latest user query only — avoids truncation and embedding
            # dilution with dense retrievers (SBERT/BERT)
            retrieval_inputs.append(user_query)
            session_memories.append(session_memory)
            user_ids.append(user_id)

        # Stage 1: Batch retrieval — fetch larger pool when reranker is active
        retrieval_topk = 100 if self.reranker else 20
        if hasattr(self.retrieval, 'batch_text_to_item_retrieval'):
            batch_retrieval_items = self.retrieval.batch_text_to_item_retrieval(retrieval_inputs, topk=retrieval_topk)
        else:
            batch_retrieval_items = [self.retrieval.text_to_item_retrieval(inp, topk=retrieval_topk) for inp in retrieval_inputs]

        # Stage 1b: Optional user-profile reranking
        if self.reranker:
            user_profiles = [self._get_user_profile_str(uid) for uid in user_ids]
            batch_retrieval_items = self.reranker.batch_rerank(user_profiles, batch_retrieval_items, topk=20)
            # batch_rerank falls back to original order when profile is None

        recommend_items = [self.item_db.id_to_metadata(items[0]) for items in batch_retrieval_items]

        # Stage 2: Batch response generation
        if hasattr(self.lm, 'batch_response_generation'):
            responses = self.lm.batch_response_generation(sys_prompts, session_memories, recommend_items)
        else:
            responses = [self.lm.response_generation(sys_prompts[i], session_memories[i], recommend_items[i])
                        for i in range(len(batch_data))]

        # Prepare results
        results = []
        for i, data in enumerate(batch_data):
            results.append({
                "user_id": data.get('user_id'),
                "user_query": data['user_query'],
                "retrieval_items": batch_retrieval_items[i],
                "recommend_item": recommend_items[i],
                "response": responses[i],
            })

        return results
