import streamlit as st

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="RAG Chatbot Tester",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

import requests
import json
import asyncio
import aiohttp
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
from typing import List, Dict, Any

# Import your RAG function (make sure the path is correct)
try:
    from main import rag_chat_function, RAGCustomerServiceBot

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Custom CSS for better styling
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e1e5e9;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: 20%;
    }
    .bot-message {
        background-color: #f8f9fa;
        color: #333;
        margin-right: 20%;
    }
    .sources {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px solid #e1e5e9;
    }
    .metrics {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"streamlit_test_{int(time.time())}"
    if 'rag_bot' not in st.session_state and RAG_AVAILABLE:
        st.session_state.rag_bot = RAGCustomerServiceBot()
    if 'response_times' not in st.session_state:
        st.session_state.response_times = []
    if 'test_scenarios' not in st.session_state:
        st.session_state.test_scenarios = [
            "What are the applications of generative AI?",
            "How do I reset my password?",
            "What is your refund policy?",
            "Can you explain the pricing structure?",
            "How do I contact customer support?",
            "What services do you offer?",
            "How do I cancel my subscription?",
            "What are the system requirements?",
        ]


async def get_rag_response_async(question: str, session_id: str) -> Dict[str, Any]:
    """Get response from RAG system asynchronously"""
    try:
        if RAG_AVAILABLE:
            response = await rag_chat_function(question, session_id)
            print(response)
            return response
        else:
            return {
                "answer": "RAG system not available. Please check your setup.",
                "sources": [],
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }


def get_rag_response_sync(question: str, session_id: str) -> Dict[str, Any]:
    """Synchronous wrapper for RAG response"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(get_rag_response_async(question, session_id))
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "sources": [],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }


def test_api_endpoint(question: str, session_id: str, api_url: str) -> Dict[str, Any]:
    """Test the RAG API endpoint"""
    try:
        payload = {
            "message": question,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }

        start_time = time.time()
        response = requests.post(f"{api_url}/api/chat", json=payload, timeout=30)
        end_time = time.time()

        if response.status_code == 200:
            result = response.json()
            result['response_time'] = end_time - start_time
            result['status'] = 'success'
            return result
        else:
            return {
                "response": f"API Error: {response.status_code} - {response.text}",
                "sources": [],
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "response_time": end_time - start_time,
                "status": "error"
            }
    except Exception as e:
        return {
            "response": f"Connection Error: {str(e)}",
            "sources": [],
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "response_time": 0,
            "status": "error"
        }


def display_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>You:</strong> {message['text']}
            <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">
                {message.get('timestamp', '')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sources_html = ""
        if message.get('sources') and len(message['sources']) > 0:
            sources_html = f"""
            <div class="sources">
                📚 <strong>Sources:</strong> {', '.join(message['sources'])}
            </div>
            """

        response_time_html = ""
        if message.get('response_time'):
            response_time_html = f" (⏱️ {message['response_time']:.2f}s)"

        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 AI Assistant:</strong> {message['text']}
            {sources_html}
            <div style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.8;">
                {message.get('timestamp', '')}{response_time_html}
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    initialize_session_state()

    # Check RAG availability and show error if needed
    if not RAG_AVAILABLE:
        st.error("⚠️ RAG chatbot service not found. Make sure rag_chatbot_service.py is in the same directory.")

    # Main title
    st.title("🤖 RAG Chatbot Testing Interface")
    st.markdown("---")

    # Sidebar for configuration and testing options
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Test mode selection
        test_mode = st.selectbox(
            "Select Test Mode:",
            ["Direct Function Call", "API Endpoint", "Batch Testing"]
        )

        if test_mode == "API Endpoint":
            api_url = st.text_input("API URL:", value="http://localhost:8001")

        # Session management
        st.markdown("### 📝 Session Management")
        current_session = st.text_input("Session ID:", value=st.session_state.session_id)
        if current_session != st.session_state.session_id:
            st.session_state.session_id = current_session

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.response_times = []
            st.success("Chat history cleared!")

        # Quick test scenarios
        st.markdown("### 🚀 Quick Test Scenarios")
        for i, scenario in enumerate(st.session_state.test_scenarios):
            if st.button(f"Test: {scenario[:30]}...", key=f"scenario_{i}"):
                st.session_state.test_question = scenario

        # Performance metrics
        if st.session_state.response_times:
            st.markdown("### 📊 Performance Metrics")
            avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.metric("Average Response Time", f"{avg_time:.2f}s")
            st.metric("Total Messages", len(st.session_state.messages))

    # Main chat interface
    col1, col2 = st.columns([3, 1])

    with col1:
        st.header("💬 Chat Interface")

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                display_message(message, message.get('is_user', False))

        # Input form
        with st.form("chat_form", clear_on_submit=True):
            col_input, col_button = st.columns([4, 1])

            with col_input:
                user_input = st.text_area(
                    "Your Message:",
                    height=100,
                    placeholder="Type your question here...",
                    key="user_input"
                )

            with col_button:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                submit_button = st.form_submit_button("Send 📤", use_container_width=True)

        # Handle form submission or quick test
        question_to_process = None
        if submit_button and user_input:
            question_to_process = user_input
        elif hasattr(st.session_state, 'test_question'):
            question_to_process = st.session_state.test_question
            delattr(st.session_state, 'test_question')

        if question_to_process:
            # Add user message
            user_message = {
                'text': question_to_process,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'is_user': True
            }
            st.session_state.messages.append(user_message)

            # Show thinking indicator
            with st.spinner("🤔 AI is thinking..."):
                start_time = time.time()

                if test_mode == "Direct Function Call":
                    response = get_rag_response_sync(question_to_process, st.session_state.session_id)
                    bot_response = {
                        'text': response.get('answer', 'No response'),
                        'sources': response.get('sources', []),
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'response_time': time.time() - start_time,
                        'is_user': False
                    }

                elif test_mode == "API Endpoint":
                    response = test_api_endpoint(question_to_process, st.session_state.session_id, api_url)
                    bot_response = {
                        'text': response.get('response', response.get('answer', 'No response')),
                        'sources': response.get('sources', []),
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'response_time': response.get('response_time', 0),
                        'is_user': False
                    }

                # Record response time
                st.session_state.response_times.append(bot_response['response_time'])
                st.session_state.messages.append(bot_response)

            # Refresh the page to show new messages
            st.rerun()

    with col2:
        st.header("🔍 Testing Tools")

        # Batch testing
        if test_mode == "Batch Testing":
            st.subheader("📋 Batch Test")

            if st.button("🚀 Run All Test Scenarios", use_container_width=True):
                results = []
                progress_bar = st.progress(0)

                for i, scenario in enumerate(st.session_state.test_scenarios):
                    progress_bar.progress((i + 1) / len(st.session_state.test_scenarios))

                    start_time = time.time()
                    response = get_rag_response_sync(scenario, f"batch_test_{i}")
                    end_time = time.time()

                    results.append({
                        'Question': scenario,
                        'Response Length': len(response.get('answer', '')),
                        'Sources Count': len(response.get('sources', [])),
                        'Response Time': end_time - start_time,
                        'Status': response.get('status', 'unknown')
                    })

                # Display results
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                # Create performance chart
                fig = px.bar(df, x='Question', y='Response Time',
                             title='Response Times by Question',
                             labels={'Question': 'Test Questions', 'Response Time': 'Time (seconds)'})
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

        # System status
        st.subheader("📊 System Status")

        if RAG_AVAILABLE:
            st.success("✅ RAG System: Available")
        else:
            st.error("❌ RAG System: Unavailable")

        # Test connectivity
        if test_mode == "API Endpoint":
            if st.button("🔌 Test API Connection"):
                try:
                    response = requests.get(f"{api_url}/api/health", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ API: Connected")
                    else:
                        st.error(f"❌ API: Error {response.status_code}")
                except Exception as e:
                    st.error(f"❌ API: Connection failed - {str(e)}")

        # Export options
        st.subheader("💾 Export Options")

        if st.session_state.messages:
            # Export chat history
            export_data = {
                'session_id': st.session_state.session_id,
                'messages': st.session_state.messages,
                'performance': {
                    'total_messages': len(st.session_state.messages),
                    'avg_response_time': sum(st.session_state.response_times) / len(
                        st.session_state.response_times) if st.session_state.response_times else 0,
                    'response_times': st.session_state.response_times
                }
            }

            st.download_button(
                label="📥 Download Chat History",
                data=json.dumps(export_data, indent=2),
                file_name=f"chat_history_{st.session_state.session_id}.json",
                mime="application/json",
                use_container_width=True
            )

    # Footer with instructions
    st.markdown("---")
    st.markdown("""
    ### 📖 How to Use This Tester

    1. **Direct Function Call**: Tests the RAG function directly (fastest, best for development)
    2. **API Endpoint**: Tests the REST API endpoint (good for integration testing)
    3. **Batch Testing**: Runs multiple test scenarios automatically

    **Tips for Testing:**
    - Use different session IDs to test conversation memory
    - Try various question types (factual, procedural, conversational)
    - Monitor response times and sources
    - Export chat history for analysis
    """)


if __name__ == "__main__":
    main()