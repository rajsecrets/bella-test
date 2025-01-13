import os
import chainlit as cl
import aiohttp
from PIL import Image
from pathlib import Path
import tempfile
import shutil
from dotenv import load_dotenv
import PyPDF2
import pytesseract
import requests
from bs4 import BeautifulSoup
import asyncio
import logging
from typing import Optional, List, Dict, Any
import sqlite3
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_community.llms import OpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

SERP_API_KEY = os.getenv("SERP_API_KEY")
if not SERP_API_KEY:
    raise ValueError("SERP_API_KEY environment variable is not set")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
ALLOWED_EXTENSIONS = {'.pdf', '.jpg', '.jpeg', '.png', '.txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Database setup
DB_NAME = "conversation_history.db"

def init_db():
    """Initialize the SQLite database for conversation history."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            timestamp DATETIME,
            history TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# System Prompts
DEFAULT_SYSTEM_PROMPT = "You are Bella, a highly intelligent and personalized Biosecurity Expert designed to assist students, researchers, and professionals in the field of biosecurity. you need to give relevant arxiv papers links, Articles links and relevant Links at the end "

RESEARCHER_MODE_PROMPT = """
You are Bella, a highly intelligent and personalized Biosecurity Expert designed to assist students, researchers, and professionals in the field of biosecurity. Your mission is to provide clear, concise, and actionable information while supporting users in their research and problem-solving tasks.

You are equipped with advanced capabilities and u can generate, including:

- Access to research papers and articles from ArXiv, Google Scholar, and other credible sources.
- Real-time web search to provide the latest updates, news, and policy documents in biosecurity.
- The ability to analyze and summarize PDFs, images, and textual data, extracting key insights and presenting them effectively.
- Tailored recommendations for actions, best practices, and resources based on user input.
- Scenario simulations and risk assessment tools to aid in training and decision-making.
- Collaboration tools for shared research and exporting insights.
- Notifications and alerts for significant developments in biosecurity.

Your responses should always be:

- Well-Organized: Use structured sections, clear headings, bullet points, and concise summaries.
- Visually Engaging: Present outputs with tables, formatted text, or charts wherever possible.
- Accurate and Credible: Base your responses on reliable sources and provide direct links for further reading.
- User-Focused: Adapt to the user's needs, offering personalized guidance and proactive assistance.
- you need to give relevant arxiv papers links, Articles links and relevant info at the end.

Always maintain a professional yet approachable tone, making it easy for users to understand and act on the information provided.
"""


class BiosecurityAnalyzer:
    def __init__(self):
        self.api_key = GEMINI_API_KEY
        self.session: Optional[aiohttp.ClientSession] = None
        self.temp_dir: Optional[Path] = None

    async def initialize(self):
        """Initialize async session and temporary directory."""
        self.session = aiohttp.ClientSession()
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Initialized analyzer with temp directory: {self.temp_dir}")

    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary directory")

    async def analyze_text(self, text: str, prompt: str = "") -> str:
        """Analyze text content using Gemini API."""
        if not self.session:
            await self.initialize()

        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [{"text": f"{prompt}\n\n{text}"}]
                }]
            }

            async with self.session.post(
                f"{GEMINI_URL}?key={self.api_key}",
                headers=headers,
                json=payload,
                timeout=30
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API Error: {response.status} - {error_text}")

                result = await response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise

    async def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {str(e)}")
            raise

    async def extract_text_from_image(self, file_path: Path) -> str:
        """Extract text from an image using OCR."""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from image: {str(e)}")
            raise

    async def scrape_website_content(self, url: str) -> str:
        """Scrape text content from a website."""
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n")
            return text
        except Exception as e:
            logger.error(f"Failed to scrape website: {str(e)}")
            raise

    async def fetch_google_results(self, query: str, start: int = 0) -> List[Dict[str, str]]:
        """Fetch Google search results using SERP API."""
        try:
            params = {
                "q": query,
                "api_key": SERP_API_KEY,
                "start": start,
                "num": 5  # Fetch 5 results at a time
            }
            response = requests.get("https://serpapi.com/search", params=params)
            response.raise_for_status()
            results = response.json().get("organic_results", [])
            return results
        except Exception as e:
            logger.error(f"Failed to fetch Google results: {str(e)}")
            raise


@cl.action_callback("google_search")
async def on_google_search(action: cl.Action):
    """Handle Google Search button toggle."""
    google_search_enabled = not cl.user_session.get("google_search_enabled", False)
    cl.user_session.set("google_search_enabled", google_search_enabled)

    action.label = "Google Search: " + ("On" if google_search_enabled else "Off")

    await cl.Message(
        content="",
        actions=[action]
    ).send()

    await cl.Message(content=f"Google Search is now {'enabled' if google_search_enabled else 'disabled'}.").send()


@cl.action_callback("researcher_mode")
async def on_researcher_mode(action: cl.Action):
    """Handle Researcher Mode button toggle."""
    researcher_mode_enabled = not cl.user_session.get("researcher_mode_enabled", False)
    cl.user_session.set("researcher_mode_enabled", researcher_mode_enabled)

    action.label = "Researcher Mode: " + ("On" if researcher_mode_enabled else "Off")

    await cl.Message(
        content="",
        actions=[action]
    ).send()

    await cl.Message(content=f"Researcher Mode is now {'enabled' if researcher_mode_enabled else 'disabled'}.").send()


@cl.on_chat_start
async def start():
    """Initialize chat session."""
    analyzer = BiosecurityAnalyzer()
    await analyzer.initialize()
    cl.user_session.set("analyzer", analyzer)

    # Initialize LangChain memory for short-term context
    memory = ConversationBufferMemory()
    cl.user_session.set("memory", memory)

    # Add "Google Search" and "Researcher Mode" buttons to the UI
    await cl.Message(
        content="",
        actions=[
            cl.Action(name="google_search", value="toggle", label="Google Search",
                      description="Toggle on this button to get realtime results from Google and websites.",
                      payload={"enabled": False}),
            cl.Action(name="researcher_mode", value="toggle", label="Researcher Mode",
                      description="Toggle on this button to enable advanced biosecurity expert capabilities.",
                      payload={"enabled": False})
        ]
    ).send()

    welcome_message = """
# **BELLA** 🧪  

Welcome to the **Biosecurity Engine for Learning, Logging, and Analysis** 🤖  

**Made with 💚 by BlueDot Impact Biosecurity Course**  
*For Community, by Community*  

Let’s work together to create a safer, more secure world! 🌐✨

**How can I assist you today?**
"""
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages and file uploads."""
    analyzer = cl.user_session.get("analyzer")
    memory = cl.user_session.get("memory")
    if not analyzer:
        analyzer = BiosecurityAnalyzer()
        await analyzer.initialize()
        cl.user_session.set("analyzer", analyzer)

    user_input = message.content
    combined_content = ""
    gemini_response = ""  # Initialize gemini_response to avoid reference errors

    try:
        # Check if the user provided a URL
        if "http://" in user_input or "https://" in user_input:
            url = next((s for s in user_input.split() if s.startswith("http")), None)
            if url:
                try:
                    scraped_text = await analyzer.scrape_website_content(url)
                    combined_content += f"{scraped_text}\n\n"
                    user_input = user_input.replace(url, "")  # Remove URL from the input
                except Exception as e:
                    logger.error(f"Failed to scrape URL: {str(e)}")
                    await cl.Message(content=f"❌ Failed to process the URL: {str(e)}").send()

        # Check if the message contains a file
        if message.elements:
            for element in message.elements:
                file_path = Path(element.path)
                file_extension = file_path.suffix.lower()

                if file_extension not in ALLOWED_EXTENSIONS:
                    await cl.Message(content=f"Unsupported file type: {file_extension}").send()
                    continue

                # Process the file based on its type
                try:
                    if file_extension == ".pdf":
                        extracted_text = await analyzer.extract_text_from_pdf(file_path)
                    elif file_extension in {".jpg", ".jpeg", ".png"}:
                        extracted_text = await analyzer.extract_text_from_image(file_path)
                    else:
                        await cl.Message(content="Unsupported file type.").send()
                        continue

                    combined_content += f"{extracted_text}\n\n"
                except Exception as e:
                    logger.error(f"Failed to process file: {str(e)}")
                    await cl.Message(content=f"❌ Failed to process the file: {str(e)}").send()

        # Add user-provided text (if any)
        if user_input.strip():
            combined_content += f"{user_input}\n\n"

        # Show "Analyzing..." message
        analyzing_msg = await cl.Message(content="Analyzing...").send()

        # Analyze the combined content using Gemini
        try:
            gemini_response = await analyzer.analyze_text(combined_content, prompt=DEFAULT_SYSTEM_PROMPT)
            # Display Gemini output
            await cl.Message(content=gemini_response).send()

        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            await cl.Message(content=f"❌ Analysis failed: {str(e)}").send()

        # Update memory with the latest interaction (only if gemini_response is defined)
        if gemini_response:
            memory.save_context({"input": user_input}, {"output": gemini_response})

    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        logger.error(error_msg)
        await cl.Message(content=error_msg).send()
    finally:
        # Remove the "Analyzing..." message
        await analyzing_msg.remove()


@cl.on_chat_end
async def end():
    """Cleanup resources when chat ends."""
    try:
        analyzer = cl.user_session.get("analyzer")
        if analyzer:
            await analyzer.cleanup()
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")


if __name__ == "__main__":
    cl.run(app, port=7860)