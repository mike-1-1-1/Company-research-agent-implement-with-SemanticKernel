# from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel

# from typing import Annotated

# from dataclasses import dataclass


# @vectorstoremodel
# @dataclass
# class ChatMessageRecord:
#     id: Annotated[str, VectorStoreField("key")]
#     role: Annotated[str, VectorStoreField("data")] = ""
#     content: Annotated[str, VectorStoreField("data")] = ""
#     timestamp: Annotated[str, VectorStoreField("data")] = ""
#     embedding: Annotated[
#         list[float], 
#         VectorStoreField(
#             "vector", 
#             dimensions=1536, 
#             distance_function="cosine",
#             #embedding_generator="content" #TODO: this might be wrong
#         )
#     ] = None
# from dataclasses import dataclass, field
# from typing import Annotated
# from semantic_kernel.data.vector import (
#     VectorStoreField, 
#     vectorstoremodel,
#     EmbeddingGeneratorSettings  # Required if using automatic embeddings
# )

# @vectorstoremodel
# @dataclass
# class ChatMessageRecord:
#     id: Annotated[str, VectorStoreField("key")]
    
#     role: Annotated[str, VectorStoreField("data")] = ""
#     content: Annotated[str, VectorStoreField("data")] = ""
#     timestamp: Annotated[str, VectorStoreField("data")] = ""
    
#     # 1. Use field(default_factory=list) to handle instantiation safely
#     # 2. Correct parameter to enable automatic embedding mapping
#     embedding: Annotated[
#         list[float], 
#         VectorStoreField(
#             "vector", 
#             dimensions=1536, 
#             distance_function="cosine",
#             embedding_settings={"content": EmbeddingGeneratorSettings()} # Replaces your TODO comment
#         )
#     ] = field(default_factory=list)

# from dataclasses import dataclass
# from typing import Annotated
# from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
# from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding

# @vectorstoremodel
# @dataclass
# class ChatMessageRecord:
#     id: Annotated[str, VectorStoreField("key")]
#     role: Annotated[str, VectorStoreField("data")] = ""
#     content: Annotated[str, VectorStoreField("data")] = ""
#     timestamp: Annotated[str, VectorStoreField("data")] = ""
#     # Pass your actual embedder instance directly to the vector field
#     embedding: Annotated[
#         list[float] | str | None,
#         VectorStoreField(
#             "vector", 
#             dimensions=1536, 
#             embedding_generator=OpenAITextEmbedding(ai_model_id="text-embedding-3-small")
#         )
#     ] = None

#     # def __post_init__(self):
#     #     """Pre-processes text for automatic embedding."""
#     #     if self.embedding is None:
#     #         self.embedding = self.content # Staging content to be embedded
#     def __post_init__(self):
#         """Only pass to the auto-generator if the data is a text string."""
#         # If it's already a list of floats, DO NOT let the generator touch it
#         if isinstance(self.embedding, list):
#             return 
            
#         # If it's empty/None, map the string contents over for auto-generation
#         if self.embedding is None or self.embedding == "":
#             self.embedding = self.content

from dataclasses import dataclass
from typing import Annotated
from semantic_kernel.data.vector import DistanceFunction, VectorStoreField, vectorstoremodel

@vectorstoremodel
@dataclass
class ChatMessageRecord:
    id: Annotated[str, VectorStoreField("key")]
    role: Annotated[str, VectorStoreField("data")] = ""
    # Text property holding the message text:
    #content: Annotated[str, VectorStoreField(is_full_text_searchable=True)]
    content: Annotated[str, VectorStoreField("data", is_full_text_indexed=True)] = ""
    timestamp: Annotated[str, VectorStoreField("data")] = ""
    
    # Strictly define this as a vector storage location
    embedding: Annotated[
        list[float] | None, 
        VectorStoreField(
            "vector", 
            dimensions=1536, 
            distance_function=DistanceFunction.COSINE_DISTANCE
        )
    ] = None
