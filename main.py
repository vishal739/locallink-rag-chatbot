import os
import asyncio
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate

warnings.filterwarnings("ignore")
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    message: str
    user_id: str
    session_id: str
    timestamp: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None
    session_id: str
    timestamp: str


class RAGCustomerServiceBot:
    """Improved RAG-powered Customer Service Chatbot"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.setup_rag_chain()

    def setup_rag_chain(self):
        """Initialize the RAG chain with better retrieval and prompting"""
        try:
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.environ.get("OPENAI_API_KEY")
            )

            # Initialize vector store
            self.vectorstore = PineconeVectorStore(
                index_name=os.environ.get("INDEX_NAME"),
                embedding=self.embeddings
            )

            # Initialize chat model with better parameters
            self.chat_model = ChatOpenAI(
                openai_api_key=os.environ.get("OPENAI_API_KEY"),
                model_name="gpt-3.5-turbo",
                temperature=0.1,  # Lower temperature for more consistent responses
                verbose=False,
                max_tokens=300  # Shorter responses for chat
            )

            # Better retriever configuration
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 4,  # Get more documents
                    "score_threshold": 0.7  # Only use relevant documents
                }
            )

            # Create better prompt template
            self.create_prompt_template()

            logger.info("RAG chain initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {e}")
            raise

    def create_prompt_template(self):
        """Create a polished and support-focused prompt template"""
        self.prompt_template = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template="""You are a friendly and professional customer support assistant for **LocalLink**, a platform that connects customers with trusted local service providers.

    Use only the information provided in the context below to assist with the customer’s inquiry. If the context doesn’t include the answer, respond as a support agent would: either acknowledge the limitation and offer to escalate, or politely let them know it's outside your scope.

    ---

    🧠 **Knowledge Base Context**:
    {context}

    💬 **Chat History**:
    {chat_history}

    ❓ **Customer’s Question**:
    {question}

    ---

    📋 **Instructions**:
    - Greet users warmly if they say "hi", "hello", or similar.
    - If the question is related to customer support (e.g. booking, cancellations, provider info, issues with service), respond using the context and your tone as a helpful agent.
    - If the answer is not in the context but clearly a customer support question, say:
      *"I don’t have the exact information in our knowledge base, but I’d be happy to connect you with our support team for a quick resolution."*
    - If the question is unrelated to LocalLink or its services, politely say you’re unable to help:
      *"I’m here to help with LocalLink services. For other questions, I recommend checking with the appropriate source."*
    - Always keep your responses:
      - Supportive and concise
      - Polite and professional
      - Accurate — never guess or make up answers

    📝 **Your Answer**:
    """
        )

    def get_or_create_session(self, session_id: str) -> Dict:
        """Get existing session or create new one"""
        if session_id not in self.sessions:
            # Create memory for conversation history
            memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                output_key="answer",
                return_messages=True,
                k=5  # Keep last 5 exchanges (reduced for better context)
            )

            # Create conversational chain with custom prompt
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.chat_model,
                retriever=self.retriever,
                memory=memory,
                return_source_documents=True,
                chain_type="stuff",
                verbose=True,  # Enable for debugging
                combine_docs_chain_kwargs={
                    "prompt": self.prompt_template
                }
            )

            self.sessions[session_id] = {
                "qa_chain": qa_chain,
                "memory": memory,
                "created_at": datetime.now(),
                "message_count": 0
            }

            logger.info(f"Created new session: {session_id}")

        return self.sessions[session_id]

    def preprocess_question(self, question: str) -> str:
        """Clean and preprocess the user question"""
        # Remove extra whitespace and normalize
        question = question.strip()

        # Add question words if missing for better retrieval
        question_starters = ['how', 'what', 'where', 'when', 'why', 'can', 'do', 'does', 'is', 'are']
        first_word = question.lower().split()[0] if question.split() else ""

        if first_word not in question_starters and not question.endswith('?'):
            question = f"How do I {question.lower()}"

        return question

    async def get_response(self, question: str, session_id: str) -> ChatResponse:
        """Get response from RAG chain with improved processing"""
        try:
            session = self.get_or_create_session(session_id)
            qa_chain = session["qa_chain"]

            # Preprocess the question for better retrieval
            processed_question = self.preprocess_question(question)
            logger.info(f"Processing question: '{question}' -> '{processed_question}'")

            # Test retrieval first
            relevant_docs = self.retriever.get_relevant_documents(processed_question)
            logger.info(f"Retrieved {len(relevant_docs)} documents for question")

            if not relevant_docs:
                logger.warning("No relevant documents found")
                fallback_response = ChatResponse(
                    response="I don't have specific information about that in our knowledge base. Could you please rephrase your question or contact our support team for assistance?",
                    sources=[],
                    session_id=session_id,
                    timestamp=datetime.now().isoformat()
                )
                return fallback_response

            # Get response from chain
            result = await asyncio.get_event_loop().run_in_executor(
                None, qa_chain, {"question": processed_question}
            )

            # Extract and clean sources (remove duplicates)
            sources = []
            if "source_documents" in result:
                seen_sources = set()
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "Unknown")
                    if source not in seen_sources:
                        sources.append(source)
                        seen_sources.add(source)

            # Update session stats
            session["message_count"] += 1

            response = ChatResponse(
                response=result["answer"],
                sources=sources,  # Don't include sources in response (as requested)
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )

            logger.info(f"Generated response for session {session_id}: {response.response[:100]}...")
            return response

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_response = ChatResponse(
                response="I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team directly.",
                sources=[],
                session_id=session_id,
                timestamp=datetime.now().isoformat()
            )
            return error_response

    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get conversation history for a session"""
        if session_id not in self.sessions:
            return []

        memory = self.sessions[session_id]["memory"]
        messages = memory.chat_memory.messages

        history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({"type": "human", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"type": "ai", "content": message.content})

        return history

    def clear_session(self, session_id: str):
        """Clear a specific session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Cleared session: {session_id}")

    async def test_retrieval(self, question: str) -> Dict:
        """Test retrieval for debugging"""
        try:
            docs = self.retriever.get_relevant_documents(question)
            return {
                "question": question,
                "num_docs": len(docs),
                "docs": [{"content": doc.page_content[:200], "source": doc.metadata.get("source")} for doc in docs]
            }
        except Exception as e:
            return {"error": str(e)}


# Initialize the improved chatbot
chatbot = RAGCustomerServiceBot()

# FastAPI app
app = FastAPI(title="Improved RAG Customer Service Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket connection manager (unchanged)
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            question = message_data.get("message")
            session_id = message_data.get("session_id", client_id)

            if question:
                response = await chatbot.get_response(question, session_id)

                await manager.send_personal_message(
                    json.dumps({
                        "response": response.response,
                        "sources": [],  # Don't send sources to frontend
                        "session_id": response.session_id,
                        "timestamp": response.timestamp
                    }),
                    client_id
                )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        manager.disconnect(client_id)


# REST API endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """REST endpoint for chat"""
    try:
        response = await chatbot.get_response(message.message, message.session_id)
        # Remove sources from response
        response.sources = []
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/test-retrieval/{question}")
async def test_retrieval_endpoint(question: str):
    """Test retrieval for debugging"""
    try:
        result = await chatbot.test_retrieval(question)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/history/{session_id}")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        history = chatbot.get_session_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"History endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session"""
    try:
        chatbot.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Clear session error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# Direct function for integration with other services
async def rag_chat_function(question: str, session_id: str = "default") -> dict:
    """Direct function to get RAG chat response"""
    try:
        response = await chatbot.get_response(question, session_id)
        return {
            "answer": response.response,
            "sources": [],  # Don't include sources
            "session_id": response.session_id,
            "timestamp": response.timestamp,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"RAG chat function error: {e}")
        return {
            "answer": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
            "sources": [],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)