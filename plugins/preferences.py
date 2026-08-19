import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any
from uuid import uuid4

from semantic_kernel import Kernel
from semantic_kernel.connectors.in_memory import InMemoryStore
from semantic_kernel.functions import kernel_function
from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
from semantic_kernel.functions.kernel_arguments import KernelArguments

#TODO: ensure preferences are substituted and not accumulated

import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

#from model.UserPreferenceRecord import UserPreferenceRecord

class UserPreferencesPlugin:
    """Persist and reuse lightweight user preferences in the shared in-memory store."""

    def __init__(self):
        print("Entering UserPreferencesPlugin.__init__()")
        #self.memory = memory
        self._preferences: dict[str, str] = {}
        #self.preference_collection = None

        # if self.memory is not None:
        #     print("Initializing preference_collection in UserPreferencesPlugin.__init__()")
            # self.preference_collection = self.memory.get_collection(
            #     collection_name="user_preferences",
            #     #record_type=UserPreferenceRecord,
            #     embedding_generator=None,
            # )

    # async def ensure_ready(self) -> None:
    #     print("Entering UserPreferencesPlugin.ensure_ready()")
    #     if self.preference_collection is not None:
    #         await self.preference_collection.ensure_collection_exists()

    async def save_preference(self, preference_type: str, preference_value: str, source: str = "user_input") -> dict[str, Any]:
        print('executing save_preference()')
        normalized_type = self._normalize_preference_type(preference_type)
        normalized_value = self._normalize_preference_value(normalized_type, preference_value)

        self._preferences[normalized_type] = normalized_value

        # if self.preference_collection is not None:
        #     await self.ensure_ready()
        #     record = UserPreferenceRecord(
        #         id=str(uuid4()),
        #         preference_type=normalized_type,
        #         preference_value=normalized_value,
        #         updated_at=datetime.now(timezone.utc).isoformat(),
        #         source=source,
        #     )
        #     print(f"Upserting in InMemoryStore preference_collection: [{normalized_type}]: {normalized_value[:30]}...")
        #     await self.preference_collection.upsert(record)

        return {
            "saved": True,
            "preference_type": normalized_type,
            "preference_value": normalized_value,
        }

    def _register_capture_preferences_function(self, kernel: Kernel):
        print("_register_capture_preferences_function()")
        """Register a semantic function that extracts preference hints from the user's message."""
        config_parameters_prompt = """
        You are a preference extraction assistant. Analyze the user's message and infer the following preferences.
        Return ONLY valid JSON with these keys:
        {
          "detail_level": "brief" | "detailed" | "balanced" | null,
          "report_format": "table" | "summary" | null,
          "industry_focus": "finance" | "technology" | "healthcare" | null
        }

        User message:
        [DATA_START]
        {{$user_input}}
        [DATA_END]
        """

        return kernel.add_function(
            function_name="CapturePreferences",
            plugin_name="UserPreferences",
            prompt=config_parameters_prompt,
            description="Extracts user preferences such as detail level, report format, and industry focus from a message.",
        )

    # @kernel_function(
    #     name="CapturePreference",
    #     description="Captures user preferences preferences"
    # )
    async def capture_preferences(self, kernel: Kernel, user_input: str) -> list[dict[str, Any]]:
        print('executing capture_preferences()')
        """Extract preferences from the message using a registered semantic function."""
        preference_function = self._register_capture_preferences_function(kernel) # TODO: debug if this like actually repeating 
        arguments = KernelArguments(user_input=user_input)
        result = await kernel.invoke(
            function=preference_function,
            arguments=arguments,
            plugin_name="UserPreferences",
            function_name="CapturePreferences",
        )

        print('result (UserPreferences, CapturePreferences):', result)

        #TODO: add result json validation
        # raw_output = ""
        # if result is not None:
        #     if hasattr(result, "value"):
        #         raw_output = str(result.value)
        #     elif hasattr(result, "result"):
        #         raw_output = str(result.result)
        #     else:
        #         raw_output = str(result)

        parsed_preferences = self._parse_preference_payload(result)

        saved: list[dict[str, Any]] = []

        for preference_key, preference_value in parsed_preferences.items():
            if not preference_value:
                continue
            if preference_key == "detail_level":
                saved.append(await self.save_preference("Preferred detail level", preference_value))
            elif preference_key == "report_format":
                saved.append(await self.save_preference("Report format", preference_value))
            elif preference_key == "industry_focus":
                saved.append(await self.save_preference("Industry focus", preference_value))
        print('saved: ', saved)

        return saved

    # @kernel_function(
    #     name="GetPreferences",
    #     description="Gets stored user preferences"
    # )
    # def get_preferences(self) -> dict[str, Any]:
    #     """Return the current in-memory preferences as a dictionary."""
    #     print('executing get_preferences()')
    #     print('self._preferences: ', self._preferences)
    #     return {"preferences": dict(self._preferences)}

    def build_context_prompt(self) -> str:
        print("Entering UserPreferencesPlugin.build_context_prompt()")
        if not self._preferences:
            return ""

        lines = ["User preferences to honor in your response:"]
        for preference_type, preference_value in self._preferences.items():
            display_name = preference_type.replace("_", " ").title()
            lines.append(f"- {display_name}: {preference_value}")

        return "\n".join(lines)

    def build_context_note(self) -> str:
        print("build_context_note")
        if not self._preferences:
            print("warning: not preferences")
            return ""

        preference_snippets = []
        for preference_type, preference_value in self._preferences.items():
            display_name = preference_type.replace("_", " ").title()
            preference_snippets.append(f"{display_name}: {preference_value}")

        joined_preferences = "; ".join(preference_snippets)
        return f"I’m tailoring this response using your stored preferences: {joined_preferences}."

    def _parse_preference_payload(self, result: any) -> dict[str, Any]:
        print("Entering UserPreferencesPlugin._parse_preference_payload()")
        # Initialize the final dictionary to store merged data
        data = {}

        # 1. Loop through every element in value
        for item in result.value:
            # Safely ensure the item and its content exist
            if not hasattr(item, "content") or not item.content:
                continue

            # 2. Search for the JSON block in the current content
            match = re.search(r"```json\s*(.*?)\s*```", item.content, re.DOTALL)

            if match:
                try:
                    json_string = match.group(1)
                    # 3. Parse the JSON string
                    candidate_data = json.loads(json_string)

                    # 4. Join/Merge the parsed data into final_data
                    data.update(candidate_data)

                except json.JSONDecodeError:
                    print(f"Skipping: Found a JSON block but it contains invalid syntax.")
            else:
                print("No valid JSON markdown block found in this element.")

        # Output the combined results
        print("Final Merged Dictionary:", data)
        return data

    def _normalize_preference_type(self, preference_type: str) -> str:
        print("Entering UserPreferencesPlugin._normalize_preference_type()")
        normalized = (preference_type or "").strip().lower()
        if "detail" in normalized or "level" in normalized:
            return "detail_level"
        if "report" in normalized or "format" in normalized:
            return "report_format"
        if "industry" in normalized or "focus" in normalized or "sector" in normalized:
            return "industry_focus"
        return normalized.replace(" ", "_")

    def _normalize_preference_value(self, preference_type: str, preference_value: str) -> str:
        print("Entering UserPreferencesPlugin._normalize_preference_value()")
        normalized_value = (preference_value or "").strip()
        if not normalized_value:
            return normalized_value

        lowered = normalized_value.lower()
        if preference_type == "detail_level":
            if lowered in {"brief", "short", "concise", "summary"}:
                return "brief"
            if lowered in {"detailed", "deep", "comprehensive", "full"}:
                return "detailed"
            if lowered in {"balanced", "standard", "normal"}:
                return "balanced"
        if preference_type == "report_format":
            if lowered in {"table", "tabular"}:
                return "table"
            if lowered in {"summary", "bullet", "bullet points"}:
                return "summary"
        if preference_type == "industry_focus":
            return lowered

        return normalized_value
    
    # @kernel_function(
    #     name="get_user_preferences",
    #     description="Retrieves active user preferences such as detail level, report format, or industry focus."
    # )
    # async def get_user_preferences(self) -> str:
    #     """Fetches stored preferences to help format the final response."""
    #     print("Executing get_user_preferences()")
    #     #await self.preference_collection.ensure_collection_exists()

    #     # Access internal memory store dictionary directly
    #     raw_dict = getattr(self.preference_collection, "_store", {})
    #     keys = list(raw_dict.keys())

    #     if keys:
    #         # Fetch records by primary keys
    #         records = await self.preference_collection.get_batch(keys=keys)
    #         async for record in records:
    #             print(f"Record: {vars(record)}")
    #     else:
    #         print(f"No records stored in preference_collection")
        
    #     # Search or get batch from the store
    #     #results = await self.preference_collection.search(query="", top=10)
    #     # results = await self.preference_collection.search(top=10)

    #     # prefs = []
    #     # async for res in results:
    #     #     rec = res.record
    #     #     prefs.append(f"{rec.preference_type}: {rec.preference_value}")
            
    #     # if not prefs:
    #     #     return "No specific user preferences set."
            
    #     # return "Active User Preferences:\n" + "\n".join(prefs)
