from pydantic import BaseModel
import os

class Defaults(BaseModel):
    fixed_size_tokens: int = 500
    fixed_overlap_tokens: int = 80
    sliding_window: int = 400
    sliding_stride: int = 200
    top_k: int = 6
    max_sentences: int = 4

class AzureOpenAIConfig(BaseModel):
    api_key: str = os.environ.get('AZURE_OPENAI_API_KEY', '')
    endpoint: str = os.environ.get('AZURE_OPENAI_ENDPOINT', '')
    deployment: str = os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')
    api_version: str = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
    embedding_deployment: str = os.environ.get('AZURE_OPENAI_EMBEDDING', 'text-embedding-3-small')

DEFAULTS = Defaults()
AZURE_CONFIG = AzureOpenAIConfig()
