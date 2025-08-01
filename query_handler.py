from typing import Dict, Optional
from datetime import datetime

class QueryHandler:
    def __init__(self, authenticated_user: Dict, rag_system, authenticator):
        self.user = authenticated_user
        self.rag_system = rag_system
        self.authenticator = authenticator
        self.hr_contact = "hr@example.com"

    def _refresh_user_data(self):
        """Refresh user data from Google Sheets"""
        try:
            self.user = self.authenticator.get_authenticated_user()
            return True
        except Exception as e:
            print(f"Warning: Could not refresh user data: {str(e)}")
            return False

    def handle_query(self, query: str) -> str:
        """Process user queries with semantic understanding"""
        query = query.lower().strip()

        # Handle leave application explicitly
        if query.startswith("apply for leave"):
            return self._handle_leave_application(query)

        # If user asks for leave balance directly
        if any(term in query for term in ["leave balance", "remaining leaves", "how many leaves"]):
            self._refresh_user_data()
            return f"Your current leave balance: {self.user['remaining_leaves']} days"

        # Otherwise: treat it as a semantic HR policy query
        try:
            response = self.rag_system.query_policy(query)
            return self._refine_policy_response(response)
        except Exception as e:
            return f"⚠️ Sorry, there was an error handling your question: {str(e)}"

    def _handle_leave_application(self, query: str) -> str:
        """Process leave application"""
        try:
            # Parse leave days from query
            parts = query.split()
            days = float(parts[3]) if len(parts) > 3 else None

            if not days:
                return "Please specify leave days like: 'apply for leave 2.5'"

            # Apply for leave
            success = self.authenticator.apply_for_leave(self.user['username'], days)
            if success:
                # Refresh user data after successful application
                self._refresh_user_data()
                return (f"✅ Leave application for {days} days submitted successfully!\n"
                        f"Remaining leaves: {self.user['remaining_leaves']}\n"
                        f"Status: Pending approval")
        except ValueError as e:
            return f"❌ {str(e)}"
        except Exception as e:
            return f"❌ Failed to apply for leave: {str(e)}"

    def _handle_leave_query(self, query: str) -> str:
        """Handle leave-related queries"""
        self._refresh_user_data()

        if "balance" in query or "remaining" in query or "left" in query:
            return f"Your current leave balance: {self.user['remaining_leaves']} days"
        else:
            response = self.rag_system.query_policy(query)
            return self._refine_policy_response(response)

    def _refine_policy_response(self, response: str) -> str:
        """Refine policy responses with user-specific info"""
        if "leave" in response.lower() and "remaining_leaves" in self.user:
            response += f"\n\nYour current leave balance: {self.user['remaining_leaves']} days"
        return response

    def _handle_unrelated_hr_query(self, query: str) -> str:
        """Handle HR queries not directly related to leave policies"""
        return (f"For questions about benefits and perks, please contact HR at {self.hr_contact}.\n"
                "I can help with leave policies and applications.")

    def _get_default_response(self) -> str:
        """Default response for unrecognized queries"""
        return ("I'm not sure I understand your question. I can help with:\n"
                "- Leave applications and balances\n"
                "- HR policy questions\n"
                "Try rephrasing your question or contact HR for more complex queries.")
