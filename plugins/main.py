#!/usr/bin/env python3
"""
Semantic Kernel AI Agent Core
Initializes and manages an AI agent powered by Microsoft Semantic Kernel
"""
#TODO: fix that sometimes ticker is not actually used for the interaction
#TODO: investigate that sometimes datavalidation function is called multiple
#  times and analyze if it's actually repeating unnecessary validations
#TODO: summarize history to prevent excesive token input

import os
import traceback
from openai import AsyncOpenAI
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIPromptExecutionSettings, OpenAITextEmbedding
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents import ChatHistory, ChatMessageContent, AuthorRole, ChatHistorySummarizationReducer
from semantic_kernel.exceptions import ServiceResponseException, VectorStoreOperationException
from semantic_kernel.functions import KernelPlugin
from semantic_kernel.functions.kernel_arguments import KernelArguments

from semantic_kernel.connectors.in_memory import InMemoryStore
# deprecated from semantic_kernel.data import vectorstoremodel, VectorStoreRecordKeyField, VectorStoreRecordTextField
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
import sys
from pathlib import Path

from dataclasses import dataclass
from typing import Annotated

#from model.UserPreferenceRecord import UserPreferenceRecord
from model.ChatMessageRecord import ChatMessageRecord

# Add the parent directory to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from data_processing import DataProcessingPlugin
from market_analysis import MarketAnalysisPlugin
from preferences import UserPreferencesPlugin

from semantic_kernel.connectors.in_memory import InMemoryStore
#import 

from uuid import uuid4
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # This loads the variables from your .env file into os.e

#fix for when it does things like embedding the ticker as json
#/v2/aggs/ticker/{'ticker': 'AAPL'}/rangge/1/month/{'year': 2026, 'month': 1, 'day': 1}/{'year': 2026, 'month': 3, 'day': 31}

class AIAgentCore:
    """Core AI Agent powered by Semantic Kernel"""

    async def load_history_from_memory(self, system_prompt: str = None, preference_context: str = None) -> ChatHistory:
        print("Entering AIAgentCore.load_history_from_memory()")
        """Retrieves stored records from InMemoryStore and hydrates a ChatHistory object."""
        chat_history = ChatHistory()
        chat_history.messages.insert(0, system_prompt) if system_prompt else None
        chat_history.messages.insert(0, ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content=preference_context,
        )) if preference_context else None
        # if system_prompt:
        #     chat_history.add_system_message(system_prompt)

        await self.chat_collection.ensure_collection_exists()

        records = []

        # 1. Fetch records using tracked keys with self.chat_collection.get()
        if self.chat_record_keys:
            for key in self.chat_record_keys:
                try:
                    # Use get(key=...) which is guaranteed to exist on InMemoryCollection
                    record = await self.chat_collection.get(key=key)
                    if record:
                        records.append(record)
                except Exception as exc:
                    print(f"[Memory Store] Error fetching key {key}: {exc}")

        # 2. Direct internal store fallback (if no records retrieved or keys list was empty)
        if not records:
            raw_store = (
                getattr(self.chat_collection, "_store", {})
                or getattr(self.chat_collection, "inner_collection", {})
                or getattr(self.chat_collection, "_data", {})
            )
            if isinstance(raw_store, dict):
                records = list(raw_store.values())

        # Sort records chronologically by timestamp
        records.sort(key=lambda r: getattr(r, "timestamp", ""))

        print(f"[Memory Store] Successfully loaded {len(records)} records from store.")

        # 3. Hydrate ChatHistory
        for rec in records:
            role = str(getattr(rec, "role", "")).lower()
            content = getattr(rec, "content", "")

            if role == "user":
                chat_history.add_user_message(content)
            elif role == "assistant":
                chat_history.add_assistant_message(content)
            elif role == "system":
                chat_history.add_system_message(content)

        print('complete chat history:',chat_history)


        # 2. Wrap the raw messages into the reducer
        reducer_history = ChatHistorySummarizationReducer(
            messages=chat_history.messages,  # <--- Pass the raw list here
            service=self.openai_chat_service,
            #target_count=1,
            #threshold_count=1,
            target_count=8,
            threshold_count=4,
            auto_reduce=True,
            summarization_instructions="Summarize the core user requests and decisions made so far."
        )

        # 3. Explicitly trigger the reduction right away if auto_reduce didn't run
        reduced_history = await reducer_history.reduce()

        #fix that reduced history is yielding None

        print('reduced_history:', reduced_history)

        if reduced_history is not None:
            active_chat_history = reduced_history
            print("[History Reducer] Chat history was summarized and reduced.")
        else:
            active_chat_history = chat_history
            print(f"[History Reducer] Threshold not met ({len(chat_history.messages)} < {8 + 4}). Keeping full history.")

        return active_chat_history

    def __init__(self):
        """Initialize the Semantic Kernel and AI Agent"""
        print("Entering AIAgentCore.__init__()")
        self.kernel = self._initialize_kernel()
        self.settings = OpenAIPromptExecutionSettings(service_id="custom_chat_service")
        self.settings.function_choice_behavior = FunctionChoiceBehavior.Auto(auto_invoke=True)
        self.system_message = ChatMessageContent(
            role=AuthorRole.SYSTEM,
            content = 
                "You are a helpful market research assistant. Use your available tools " \
                "to fetch/validate data, generate market summarizations and generate basic trend analysis based on the user's intent." \
                "Ensure tickers you use to make fetch_data request are well formed english-alphabetic strings"
                "Always validate data right after fetching it and remember that you already validated to prevent repeated validations on same data" \
                "When doing market summarizations always do it through semantic function called market_summarization"
                "When doing trends identification do it through semantic function called basic_trends_identification"
        )
        #TODO: add chat history prunning
        #self.chat_history = ChatHistory()
        #self.chat_history.messages.insert(0, self.system_message)
        self.memory = InMemoryStore()
        self.chat_record_keys: list[str] = []  # Maintain keys created during the session
        self.preferences_plugin = UserPreferencesPlugin()
        self.embedding_service = OpenAITextEmbedding(
            service_id="embedding_service",
            ai_model_id="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.kernel.add_service(self.embedding_service)
        self.chat_collection = self.memory.get_collection(
            collection_name="user_chat_history", 
            record_type=ChatMessageRecord,
            embedding_generator=self.embedding_service
        )
 
        #seems like ai allucination #await chat_collection.create_collection_if_not_exists()
        self._setup_plugins()

    async def post_init(self):
        print("Entering AIAgentCore.post_init()")
        await self.chat_collection.ensure_collection_exists()
        #await self.preferences_plugin.ensure_ready()

    # 4. Helper function to append to ChatHistory and save to InMemoryStore
    async def persist_message_in_memory(self, role: AuthorRole, content: str):
        #embedding = await self.embedding_service.generate_embeddings(content)
        
        # Build record matching the updated dataclass fields
        record = ChatMessageRecord(
            id=str(uuid4()),
            role=role.value,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            embedding=content  # Store the raw content for now; embedding can be generated on-demand
            #embedding=embedding.flatten().tolist()
            #embedding=content
        )
        #print(f"Vector size: {len(record.embedding)}")
        # Save to store
        await self.chat_collection.ensure_collection_exists()
        try:
            print(f"Upserting in InMemoryStore chat_collection: [{role.value.upper()}]: {content[:30]}...")
            print('record: ',record)
            key = await self.chat_collection.upsert(record)
            if key:
                self.chat_record_keys.append(str(key))
                print(f"[Memory Store] Successfully upserted message key: {key}")
            #await self.print_store_contents(self.chat_collection)
            # Inspect self.chat_collection directly via _store
            #raw_store = getattr(self.chat_collection, "_store", {})
            # print(f"--- Chat Store Count: {len(raw_store)} ---")
            # for key, rec in raw_store.items():
            #     print(f"[{key}]: {rec.content}")
            # 2. Fetch the record using public API get()
            # fetched_record = await self.chat_collection.get(key=upserted_key)
            # print(f"Fetched Record from Store: {vars(fetched_record)}")
            # Search for all stored records in the collection
            #search_results = await self.chat_collection.search(query="", top=10)

            # count = 0
            # for result in search_results:
            #     count += 1
            #     print(f"[{count}] Stored Message: {result}")
                
            # print(f"--- Chat Store Count: {count} ---")
        except VectorStoreOperationException as e:
            print("\n--- CRITICAL ERROR INNER DETAILS ---")
            if hasattr(e, '__cause__') and e.__cause__:
                print(f"Root Cause Error: {e.__cause__}")
            else:
                print(f"Direct Exception Text: {str(e)}")
            print("------------------------------------\n")
            raise e
        except Exception as e:
            print(f"General upsert error: {e}")

    def generate_and_register_memory_store_plugin(self):
        print("Entering AIAgentCore.generate_and_register_memory_store_plugin()")
        #Memory store search plugin
        # 1. Generate search function
        search_function = self.chat_collection.create_search_function(
            function_name="search_history",
            description="Searches through old chat history records semantics. From latest to oldest.",
            top = 5
        )

        # 2. Instantiate KernelPlugin passing the map to the constructor
        memory_store_plugin = KernelPlugin(
            name="MemoryStore",
            description="Provides access to vector-based semantic conversation memory.",
            functions={"search_history": search_function}  # <--- Clean Python dictionary map
        )

        # 3. Add to the kernel
        self.kernel.add_plugin(memory_store_plugin)

    def _setup_plugins(self):
        """Register plugins with the kernel"""
        print("Entering AIAgentCore._setup_plugins()")
        self.data_processing_plugin = DataProcessingPlugin()
        self.kernel.add_plugin(self.data_processing_plugin, plugin_name="DataProcessingPlugin")

        self.market_analysis_plugin = MarketAnalysisPlugin()
        self.kernel.add_plugin(self.market_analysis_plugin, plugin_name="MarketAnalysisPlugin")

        #TODO: Error Message is caused by below one, need to complete the prompt: ("<class 'semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion.OpenAIChatCompletion'> service failed to complete the prompt", TypeError('Object of type _PydanticGeneralMetadata is not JSON serializable'))

        self.kernel.add_plugin(self.preferences_plugin, plugin_name="UserPreferencesPlugin")

        self.generate_and_register_memory_store_plugin()

    def _initialize_kernel(self) -> Kernel:
        """
        Initialize the Semantic Kernel with Azure OpenAI
        
        Returns:
            Kernel: Configured Semantic Kernel instance
        """
        print("Entering AIAgentCore._initialize_kernel()")
        kernel = Kernel()

        my_custom_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.openai_chat_service = OpenAIChatCompletion(
            service_id="custom_chat_service",
            ai_model_id="gpt-4o",
            async_client=my_custom_client,
        )

        kernel.add_service(self.openai_chat_service)

        return kernel

    async def _capture_preferences(self, user_input: str) -> None:
        """Persist preference hints from the user message using the semantic preference-extraction function."""
        print("Entering AIAgentCore._capture_preferences()")
        await self.preferences_plugin.capture_preferences(self.kernel, user_input)

    def _build_user_context_message(self) -> str:
        """Create a lightweight system-style note that incorporates saved preferences."""
        print("Entering AIAgentCore._build_user_context_message()")
        preference_context = self.preferences_plugin.build_context_prompt()
        print(f"Preference context: {preference_context}")
        if not preference_context:
            return ""
        return (
            f"{preference_context}\n"
            "When possible, tailor your response to these preferences and briefly mention that you are doing so."
        )

    def _build_preference_note(self) -> str:
        """Create a short user-facing note explaining that the answer is based on stored preferences."""
        print("Entering AIAgentCore._build_preference_note()")
        return self.preferences_plugin.build_context_note()

    async def invoke_agent(self, user_input: str) -> str:
        """
        Invoke the AI agent with user input
        
        Args:
            user_input (str): User query or command
            
        Returns:
            str: Agent response
        """
        chat_completion = self.kernel.get_service("custom_chat_service")

        print("going to get result")

        await self._capture_preferences(user_input)

        preference_context = self._build_user_context_message()
        # if preference_context:
        #     self.chat_history.messages.insert(0, ChatMessageContent(
        #         role=AuthorRole.SYSTEM,
        #         content=preference_context,
        #     ))

        #self.chat_history.add_user_message(user_input)
        await self.persist_message_in_memory(AuthorRole.USER, user_input)

        chat_history =await self.load_history_from_memory(system_prompt=self.system_message, preference_context=preference_context)
        print("<<<going to get result with chat_history:", chat_history, '>>>')
        # result = await chat_completion.get_chat_message_content(
        #     chat_history=chat_history,
        #     settings=self.settings,
        #     kernel = self.kernel,
        # )   
        try:
            result = await chat_completion.get_chat_message_content(
                chat_history=chat_history,
                settings=self.settings,
                kernel=self.kernel,
            )
        except ServiceResponseException as ex:
            print(f"--- Semantic Kernel Service Error ---")
            print(f"what?")
            print(f"Error Message: {ex}")
            
            # Extract internal hidden system details if available
            if hasattr(ex, "inner_exception") and ex.inner_exception:
                print(f"Inner Exception Type: {type(ex.inner_exception)}")
                print(f"Inner Details: {ex.inner_exception}")
            
            # Dump full traceback to track down exactly which line failed serialization
            print("\n--- Full Debug Traceback ---")
            traceback.print_exc()
            
            # Optional: Re-raise or set fallback behavior
            raise ex

        #self.chat_history.add_message(result)
        await self.persist_message_in_memory(AuthorRole.ASSISTANT, result.content) #WARNING: actually verify that result.role exists
        print("after persisting")
        result_content = str(result.content).strip()
        preference_note = self._build_preference_note()
        if preference_note and result_content:
            print("adding preference note...")
            result_content = f"{preference_note}\n\n{result_content}"

        return result_content

    # async def print_store_contents(self, collection):
    #     #print(f'store_contents of type: {record_type}')
    #     # Retrieve the collection associated with the record type
    #     # collection = self.memory.get_collection(
    #     #     collection_name=collection_name,
    #     #     record_type=record_type,
    #     #     embedding_generator=embedding_generator
    #     # )
        
    #     # Ensure the collection is ready
    #     await collection.ensure_collection_exists()
        
    #     # List or fetch records from the collection
    #     # (Method depends on your specific Semantic Kernel version/collection API)
    #     # async for record in await collection.get(): # or equivalent retrieval iteration
    #     #     # Print a human-readable dictionary of the record
    #     #     print(vars(record))
    #     # Access stored records directly from the in-memory dict
    #     # if hasattr(collection, "_data"):
    #     #     for record in collection._data.values():
    #     #         print(vars(record))
    #     # 1. Retrieve keys stored inside the collection
    #     keys = []
    #     if hasattr(collection, "inner_store") and isinstance(collection.inner_store, dict):
    #         print('collection.inner_store.keys(): ', collection.inner_store.keys())
    #         keys = list(collection.inner_store.keys())
    #     elif hasattr(collection, "data") and isinstance(collection.data, dict):
    #         print('collection.data.keys(): ', collection.data.keys())
    #         keys = list(collection.data.keys())
    #     else:
    #         print("Warning: Unable to access collection data directly. Ensure your Semantic Kernel version supports this operation.")
    #     # 2. Fetch records in bulk using the public VectorStore API
    #     if keys:
    #         records = await collection.get_batch(keys=keys)
    #         async for record in records:
    #             print(vars(record))
    #     else:
    #         print("No records found in collection.")

    #     #raw retrieval
    #     # 1. Retrieve internal dictionary
    #     raw_dict = getattr(collection, "inner_store", getattr(collection, "data", {}))
    #     print(f"Raw internal dictionary keys: {list(raw_dict.keys())}")

    #     # 2. Directly print items from internal dict (Bypasses get_batch)
    #     if raw_dict:
    #         for record_id, record in raw_dict.items():
    #             print(f"Record [{record_id}]:", vars(record))
    #     else:
    #         print(f"No records stored in collection yet.")
    #     #1. Try accessing internal dictionary (_store is used by Semantic Kernel InMemoryCollection)
    #     raw_dict = getattr(collection, "_store", {})
        
    #     if isinstance(raw_dict, dict) and raw_dict:
    #         for rec_id, record in raw_dict.items():
    #             print(f"  Record [{rec_id}]: {vars(record)}")
    #         return

    #     # 2. Fallback: Query all records via get_batch using keys from _store if available
    #     try:
    #         keys = list(raw_dict.keys()) if isinstance(raw_dict, dict) else []
    #         if keys:
    #             records = await collection.get_batch(keys=keys)
    #             async for record in records:
    #                 print(f"  Record: {vars(record)}")
    #             return
    #     except Exception as e:
    #         print(f"  Batch fetch failed: {e}")

    #     print(f" No records stored in collection yet.")

    async def run_agent_loop(self):
        """Run interactive agent loop"""
        print("🤖 Semantic Kernel AI Agent Started")
        print("Type 'exit' to quit\n")

        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("Agent: Goodbye! 👋")
                    break
                
                if not user_input:
                    continue

                response = await self.invoke_agent(user_input)
                print(f"Agent: {response}\n")
                #await self.print_store_contents(self.chat_collection)
                #await self.print_store_contents(self.user_preferences_collection)

            except KeyboardInterrupt:
                print("\nAgent: Session interrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"General Error: {e}")
                continue
# def _capture_preferences(self, user_input: str) -> None:
#         """Capture simple preference hints from the user's input for later use."""
#         lowered = user_input.lower()
#         if "brief" in lowered or "short" in lowered:
#             self.memory.save_user_preference("detail_level", "brief")
#         if "detailed" in lowered or "deep" in lowered:
#             self.memory.save_user_preference("detail_level", "detailed")
#         if "financial" in lowered or "finance" in lowered:
#             self.memory.save_user_preference("industry_focus", "finance")
#         if "tech" in lowered or "technology" in lowered:
#             self.memory.save_user_preference("industry_focus", "technology")
#         if "report" in lowered and "table" in lowered:
#             self.memory.save_user_preference("report_format", "table")
#         if "report" in lowered and "summary" in lowered:
#             self.memory.save_user_preference("report_format", "summary")

async def main():
    """Main entry point"""
    print("Entering main()")
    agent = AIAgentCore()
    await agent.post_init() #TODO: actually fix this to be architecturally cleaner
    await agent.run_agent_loop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())