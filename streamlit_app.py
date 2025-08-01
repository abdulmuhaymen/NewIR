import streamlit as st
import os
from datetime import datetime
import google.generativeai as genai
from config import Config
from auth import Authenticator
from data_loader import PolicyDataLoader
from rag_system import PolicyRAGSystem
from query_handler import QueryHandler

# Page configuration
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .balance-box {
        font-size: 1.2rem;
        font-weight: bold;
        text-align: center;
        padding: 0.75rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }
    .leave-balance {
        color: #28a745;
        background-color: #f8f9fa;
        border: 2px solid #28a745;
    }
    .total-balance {
        color: #0056b3;
        background-color: #eef3fa;
        border: 2px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitHRAssistant:
    def __init__(self):
        self.config = Config()
        genai.configure(api_key=self.config.GEMINI_API_KEY)

        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'authenticator' not in st.session_state:
            st.session_state.authenticator = None
        if 'query_handler' not in st.session_state:
            st.session_state.query_handler = None
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
        if 'login_attempts' not in st.session_state:
            st.session_state.login_attempts = 0
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

    def initialize_system(self):
        if st.session_state.system_initialized:
            return True

        try:
            with st.spinner("Initializing HR system..."):
                st.session_state.authenticator = Authenticator(self.config)
                document_loader = PolicyDataLoader(self.config.POLICY_PDF_PATH)
                document_loader.load_all_documents()

                rag_system = PolicyRAGSystem(
                    retriever=document_loader.get_retriever(),
                    config=self.config,
                    data_loader=document_loader
                )
                rag_system.initialize_llm()
                rag_system.setup_qa_chain()

                st.session_state.rag_system = rag_system
                st.session_state.system_initialized = True

                st.success("✅ HR system initialized successfully!")
                return True

        except Exception as e:
            st.error(f"❌ Failed to initialize system: {str(e)}")
            return False

    def login_page(self):
        st.markdown("<h1 class='main-header'>🏢 HR Policy Assistant</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #666;'>GenITeam Solutions</p>", unsafe_allow_html=True)

        if not self.initialize_system():
            return

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("### 🔐 Login")

            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit_button = st.form_submit_button("Login", use_container_width=True)

                if submit_button:
                    if not username or not password:
                        st.error("Please enter both username and password")
                        return

                    if st.session_state.login_attempts >= self.config.MAX_LOGIN_ATTEMPTS:
                        st.error("Maximum login attempts reached. Please refresh the page to try again.")
                        return

                    try:
                        with st.spinner("Authenticating..."):
                            if st.session_state.authenticator.authenticate(username, password):
                                user = st.session_state.authenticator.get_authenticated_user()
                                st.session_state.user = user
                                st.session_state.authenticated = True
                                st.session_state.login_attempts = 0

                                st.session_state.query_handler = QueryHandler(
                                    user,
                                    st.session_state.rag_system,
                                    st.session_state.authenticator
                                )

                                st.success(f"✅ Welcome, {username}!")
                                st.rerun()
                            else:
                                st.session_state.login_attempts += 1
                                remaining = self.config.MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                                st.error(f"❌ Invalid credentials. {remaining} attempts remaining.")
                    except Exception as e:
                        st.error(f"⚠️ Authentication error: {str(e)}")
                        st.session_state.login_attempts += 1

    def main_dashboard(self):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("<h1 class='main-header'>🏢 HR Dashboard</h1>", unsafe_allow_html=True)
        with col2:
            if st.button("Logout", type="secondary"):
                self.logout()

        with st.sidebar:
            st.markdown("### 👤 User Info")
            st.markdown(f"**Username:** {st.session_state.user['username']}")

            leave_balance = st.session_state.user.get('remaining_leaves', 'N/A')
            total_allotted = st.session_state.user.get('total_leaves', 30)  # default = 30

            # Optional: Display balances
            # st.markdown(f"<div class='balance-box leave-balance'>{leave_balance} Days Remaining</div>", unsafe_allow_html=True)
            # st.markdown(f"<div class='balance-box total-balance'>{total_allotted} Days Allotted</div>", unsafe_allow_html=True)

            st.markdown("### ℹ️ Quick Help")
            st.markdown("""
            **What I can help with:**
            - Check leave balance
            - Answer HR policy questions
            - Explain company benefits
            """)

        self.chat_interface()

    def chat_interface(self):
        st.markdown("### 💬 Ask HR Questions")
        st.markdown("Ask me anything about HR policies, benefits, or company procedures!")

        # Show chat history using st.chat_message
        for question, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                st.markdown(answer)

        # Chat input (auto-submits on Enter + clears input after submit)
        user_query = st.chat_input("Ask your HR question here...")

        if user_query:
            with st.spinner("Processing your question..."):
                try:
                    response = st.session_state.query_handler.handle_query(user_query)
                    st.session_state.chat_history.append((user_query, response))
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    def logout(self):
        for key in list(st.session_state.keys()):
            if key not in ['system_initialized', 'rag_system', 'authenticator']:
                del st.session_state[key]

        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.query_handler = None
        st.session_state.login_attempts = 0

        st.success("👋 Logged out successfully!")
        st.rerun()

    def run(self):
        if not st.session_state.authenticated:
            self.login_page()
        else:
            self.main_dashboard()

# Main runner
if __name__ == "__main__":
    app = StreamlitHRAssistant()
    app.run()
