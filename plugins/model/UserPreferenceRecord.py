# from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

# from typing import Annotated

# from dataclasses import dataclass

# @vectorstoremodel
# @dataclass
# class UserPreferenceRecord:
#     id: Annotated[str, VectorStoreField("key")]
#     preference_type: Annotated[str, VectorStoreField("data")]
#     preference_value: Annotated[str, VectorStoreField("data")]
#     updated_at: Annotated[str, VectorStoreField("data")]
#     source: Annotated[str | None, VectorStoreField("data")] = None