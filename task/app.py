import os
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

# OpenAI Documentation: https://platform.openai.com/docs/guides/embeddings
# Langchain OpenAI Embeddings Documentation: https://python.langchain.com/docs/integrations/text_embedding/openai/
# Langchain OpenAI Chat Documentation: https://python.langchain.com/docs/integrations/chat/openai/
# Langchain FAISS Documentation: https://python.langchain.com/docs/integrations/vectorstores/faiss/
# Langchain RecursiveCharacterTextSplitter Documentation: https://python.langchain.com/docs/how_to/recursive_text_splitter/

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.index_folder = "microwave_faiss_index"
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """Initialize the RAG system"""
        print("🔄 Initializing Microwave Manual RAG System...")
        if os.path.exists(self.index_folder):
            return FAISS.load_local(
                folder_path=self.index_folder,
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            return self._create_new_index()

    def _create_new_index(self) -> VectorStore:
        print("📖 Loading text document...")
        text_loader = TextLoader(file_path='microwave_manual.txt', encoding='utf-8')
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", "."],
            chunk_overlap=50,
            chunk_size=300
        )
        chunks = text_loader.load_and_split(splitter)
        vector_store: VectorStore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        vector_store.save_local(self.index_folder)
        return vector_store

    def retrieve_context(self, query: str, k: int = 4, score=0.3) -> str:
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        print(f"{'=' * 100}\n🔍 STEP 1: RETRIEVAL\n{'-' * 100}")
        print(f"Query: '{query}'")
        print(f"Searching for top {k} most relevant chunks with similarity score {score}:")

        documents_with_scores: list[tuple[Document, float]] = self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            score_threshold=score
        )
        context_parts = []

        for document, score in documents_with_scores:
            context_parts.append(document.page_content)
            print('\n')
            print(f"Result score: {score}")
            print(f"Document content: {document.page_content}")
            print('\n')

        print("=" * 100)
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str) -> str:
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        augmented_prompt = USER_PROMPT.format(
            context=context,
            query=query
        )

        print(f"{augmented_prompt}\n{'=' * 100}")
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str) -> str:
        print(f"\n🤖 STEP 3: GENERATION\n{'-' * 100}")

        messages = [
            (
                "system",
                SYSTEM_PROMPT
            ),
            ("human", augmented_prompt),
        ]
        response = self.llm_client.invoke(messages)
        print(f"LLM response: {response.content}")
        return response.content


def main(rag: MicrowaveRAG):
    print("🎯 Microwave RAG Assistant")

    while True:
        user_question = input("\n> ").strip()
        context = rag.retrieve_context(user_question)
        augmented_prompt = rag.augment_prompt(user_question, context)
        rag.generate_answer(augmented_prompt)


main(
    MicrowaveRAG(
        OpenAIEmbeddings(
            model='text-embedding-3-small',
            api_key=SecretStr(OPENAI_API_KEY),
        ),
        ChatOpenAI(
            temperature=0.0,
            model='gpt-4o',
            api_key=SecretStr(OPENAI_API_KEY)
        )
    )
)
