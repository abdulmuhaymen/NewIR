
from config import Config
from auth import Authenticator
from data_loader import PolicyDataLoader  # Fixed import name
from rag_system import PolicyRAGSystem
from query_handler import QueryHandler
import sys
import google.generativeai as genai


class LeavePolicyAssistant:
    def __init__(self, config):
        self.config = config
        # Configure Gemini API
        genai.configure(api_key=config.GEMINI_API_KEY)
        self.authenticator = Authenticator(config)
        self.document_loader = PolicyDataLoader(config.POLICY_PDF_PATH)  # Fixed class name
        self.rag_system = None
        self.query_handler = None

    def initialize_system(self) -> None:
        """Initialize all system components"""
        print("\nInitializing HR Leave Policy Assistant...")
        print(f"Using policy document: {self.config.POLICY_PDF_PATH}")
        print(f"Using AI model: {self.config.GEMINI_MODEL}")
        
        # Load policy document
        try:
            self.document_loader.load_all_documents()
            print("✓ HR policies processed successfully")
        except Exception as e:
            print(f"❌ Failed to load HR policies: {str(e)}")
            sys.exit(1)
        
        # Initialize RAG system with Gemini
        try:
            self.rag_system = PolicyRAGSystem(
                retriever=self.document_loader.get_retriever(),
                config=self.config
            )
            self.rag_system.initialize_llm()
            self.rag_system.setup_qa_chain()
            print(f"✓ Gemini LLM initialized (model: {self.config.GEMINI_MODEL})")
            print("✓ RAG system ready")
        except Exception as e:
            print(f"❌ Failed to initialize Gemini: {str(e)}")
            sys.exit(1)

    def authenticate_user(self) -> None:
        """Handle user authentication with retries"""
        attempts = 0

        print("\n🔐 HR System Authentication")
        print("--------------------------")

        while attempts < self.config.MAX_LOGIN_ATTEMPTS:
            username = input("Username: ").strip()
            password = input("Password: ").strip()

            try:
                if self.authenticator.authenticate(username, password):
                    user = self.authenticator.get_authenticated_user()
                    self.query_handler = QueryHandler(user, self.rag_system, self.authenticator)
                    print(f"\n✅ Authentication successful. Welcome {username}!")
                    return
                else:
                    attempts += 1
                    remaining = self.config.MAX_LOGIN_ATTEMPTS - attempts
                    print(f"❌ Invalid credentials. {remaining} attempts remaining.\n")
            except Exception as e:
                print(f"⚠️ Authentication error: {str(e)}")
                attempts += 1

        print("\n🚫 Maximum login attempts reached. Exiting.")
        sys.exit(0)

    def run(self) -> None:
        """Main application loop"""
        self.initialize_system()
        self.authenticate_user()

        print("\nHow can I help you today?")
        print("• Check your leave balance")
        print("• Apply for leave (e.g., 'apply for leave 2')")
        print("• Ask about HR policies")
        print("• Type 'exit' to quit\n")

        while True:
            try:
                query = input("HR Query > ").strip()

                if query.lower() in ['exit', 'quit']:
                    print("\n👋 Goodbye! Have a great day!")
                    break

                if not query:
                    continue

                response = self.query_handler.handle_query(query)
                print(f"\n{response}\n")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n⚠️ Error: {str(e)}\n")


if __name__ == "__main__":
    config = Config()
    assistant = LeavePolicyAssistant(config)
    assistant.run()
