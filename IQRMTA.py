# Corrected placement for set_page_config
import streamlit as st
st.set_page_config(layout="wide")

import json
import os
import re
from openai import OpenAI

# --- Configuration ---
JSON_FILE = "iqrm_chapter_summaries.json"
API_KEY = "sk-e5UCdVT-knC7iEtrKwccstG3yZrf_i-hrKJ4dv-QpsT3BlbkFJr8PMyVYX7MJ7XipoyCL8HbAtYrbikPRvaioYVwkakA"

# --- OpenAI Client ---
try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    # Use st.warning for non-blocking errors during setup
    st.warning(f"Failed to initialize OpenAI client: {e}. Q&A feature will be disabled.")
    client = None

# --- Load Knowledge Base ---
@st.cache_data # Cache the data to avoid reloading on every interaction
def load_knowledge_base(file_path):
    if not os.path.exists(file_path):
        return None, f"Error: Knowledge base file not found at {file_path}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Sort chapters numerically
            sorted_chapters = sorted(
                [key for key in data if key.startswith('chapter_') and key.split('_')[1].isdigit()],
                key=lambda x: int(x.split('_')[1])
            )
            # Create an ordered dictionary or just return the sorted keys and original data
            return {ch: data[ch] for ch in sorted_chapters}, None
    except json.JSONDecodeError:
        return None, f"Error: Could not decode JSON from {file_path}"
    except Exception as e:
        return None, f"Error loading knowledge base: {e}"

knowledge_base, load_error = load_knowledge_base(JSON_FILE)

# --- Helper Function for OpenAI Call ---
def get_ai_response(question, context):
    if not client:
        return "Error: OpenAI client not initialized or API key invalid."
    try:
        prompt = f"""
You are an AI Teaching Assistant for Quantitative Risk Management.
Based *only* on the following context about the topic, answer the user's question concisely.
If the answer cannot be found in the context, state that clearly.

Context:
{context}

User Question: {question}

Answer:
"""
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant specializing in Quantitative Risk Management, answering questions based *only* on the provided context."}, 
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, 
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

# --- URL Parameter Handling ---
def get_topic_from_url(kb):
    params = st.query_params
    info_param = params.get("info", [None])[0]
    target_chapter_key = None
    target_topic_name = None
    error_message = None

    if info_param:
        # Expected format: chapter<number>-<TopicName>
        match = re.match(r"chapter(\d+)-(.+)", info_param, re.IGNORECASE)
        if match:
            chapter_num = match.group(1)
            topic_name_url = match.group(2).replace("-", " ") # Replace hyphens back to spaces if needed
            potential_chapter_key = f"chapter_{chapter_num}"
            
            if kb and potential_chapter_key in kb:
                chapter_data = kb[potential_chapter_key]
                # Find the topic name (case-insensitive matching might be needed)
                found_topic = None
                for topic in chapter_data:
                    # Simple case-insensitive match, might need refinement for complex names
                    if topic.lower() == topic_name_url.lower():
                        found_topic = topic
                        break
                
                if found_topic:
                    target_chapter_key = potential_chapter_key
                    target_topic_name = found_topic
                else:
                    error_message = f"Topic 	'{topic_name_url}'	 not found in Chapter {chapter_num}."
            else:
                error_message = f"Chapter {chapter_num} not found in the knowledge base."
        else:
            error_message = f"Invalid 'info' parameter format: '{info_param}'. Expected format: chapter<number>-<TopicName>."
            
    return target_chapter_key, target_topic_name, error_message

# --- Initialize Session State --- 
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_topic_context" not in st.session_state:
    st.session_state.current_topic_context = ""
if "selected_chapter_key" not in st.session_state:
    st.session_state.selected_chapter_key = None
if "selected_topic_name" not in st.session_state:
    st.session_state.selected_topic_name = None

# --- Main App Logic ---
st.title("IQRM AI Teaching Assistant")

# Display loading errors if any
if load_error:
    st.error(load_error)
    st.stop() # Stop execution if KB didn't load

if not knowledge_base:
    st.warning("Knowledge base is empty.")
    st.stop()

# Get target chapter/topic from URL, but only on first load or if state is not set
url_chapter_key, url_topic_name, url_error = get_topic_from_url(knowledge_base)

# Show URL error if present
if url_error:
    st.warning(url_error)

# Determine initial selection: URL param > session state > first item
initial_chapter_key = url_chapter_key or st.session_state.selected_chapter_key or list(knowledge_base.keys())[0]
initial_topic_name = None
initial_chapter_index = list(knowledge_base.keys()).index(initial_chapter_key)

if initial_chapter_key:
    chapter_data = knowledge_base.get(initial_chapter_key, {})
    topic_options = list(chapter_data.keys())
    if topic_options:
        # If URL provided a topic, use it, otherwise use session state or default to first topic
        potential_initial_topic = url_topic_name if url_chapter_key == initial_chapter_key else st.session_state.selected_topic_name
        if potential_initial_topic and potential_initial_topic in topic_options:
            initial_topic_name = potential_initial_topic
        else:
            initial_topic_name = topic_options[0] # Default to first topic if URL/session state invalid

initial_topic_index = topic_options.index(initial_topic_name) if initial_topic_name and topic_options else 0

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")
chapter_options = list(knowledge_base.keys())
chapter_display_names = [f"Chapter {key.split('_')[1]}" for key in chapter_options]

# Use the determined initial index for chapter selection
selected_chapter_index = st.sidebar.selectbox(
    "Select Chapter:", 
    range(len(chapter_options)), 
    index=initial_chapter_index,
    format_func=lambda i: chapter_display_names[i],
    key="chapter_select"
)
selected_chapter_key = chapter_options[selected_chapter_index]

# Update chapter in session state
st.session_state.selected_chapter_key = selected_chapter_key

# Topic Selection within the selected chapter
chapter_data = knowledge_base.get(selected_chapter_key, {})
topic_options = list(chapter_data.keys())

if not topic_options:
    st.sidebar.warning(f"No topics found for Chapter {selected_chapter_key.split('_')[1]}.")
    selected_topic_name = None
else:
    # Determine topic index: if chapter changed, use 0, else use initial_topic_index if chapter matches initial
    current_topic_index = initial_topic_index if selected_chapter_key == initial_chapter_key else 0
    
    selected_topic_index_from_widget = st.sidebar.selectbox(
        "Select Topic:", 
        range(len(topic_options)), 
        index=current_topic_index, 
        format_func=lambda i: topic_options[i],
        key="topic_select"
    )
    selected_topic_name = topic_options[selected_topic_index_from_widget]
    
    # Update topic in session state
    st.session_state.selected_topic_name = selected_topic_name

# --- Main Area --- 
if selected_chapter_key and selected_topic_name:
    st.header(f"Chapter {selected_chapter_key.split('_')[1]}: {selected_topic_name}")
    
    topic_data = knowledge_base[selected_chapter_key][selected_topic_name]
    
    # Update topic context in session state when topic changes
    new_context = f"Topic: {selected_topic_name}\nSummary: {topic_data.get('summary', '')}\nSubtitles: {', '.join(topic_data.get('subtitles', []))}"
    # Check if context *or* selected topic name has changed to reset chat
    if st.session_state.current_topic_context != new_context or st.session_state.get('_previous_topic') != selected_topic_name:
        st.session_state.current_topic_context = new_context
        st.session_state.messages = [] # Clear chat history when topic changes
        st.session_state._previous_topic = selected_topic_name # Track topic change
        # Use st.experimental_rerun() for older Streamlit versions if needed
        st.rerun() 

    # Display Topic Content in an expander
    # Default to expanded=True if loaded via URL parameter for the first time
    expand_details = url_topic_name == selected_topic_name and url_chapter_key == selected_chapter_key
    with st.expander("View Topic Details (Summary, Subtitles, Quiz, Tasks)", expanded=expand_details):
        st.subheader("Summary")
        st.write(topic_data.get("summary", "Summary not available."))
        
        st.subheader("Subtitles (H4)")
        subtitles = topic_data.get("subtitles", [])
        if subtitles:
            for sub in subtitles:
                st.markdown(f"- {sub}")
        else:
            st.write("No specific subtitles listed for this topic.")

        st.subheader("Quiz Questions")
        quiz_questions = topic_data.get("quiz_questions", [])
        if quiz_questions:
            for i, q in enumerate(quiz_questions):
                st.markdown(f"{i+1}. {q}")
        else:
            st.write("No quiz questions available.")

        st.subheader("Training Tasks")
        training_tasks = topic_data.get("training_tasks", [])
        if training_tasks:
            for i, t in enumerate(training_tasks):
                st.markdown(f"{i+1}. {t}")
        else:
            st.write("No training tasks available.")

    # Chat Interface
    st.divider()
    st.subheader(f"Chat about: {selected_topic_name}")

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input(f"Ask a question about {selected_topic_name}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = get_ai_response(prompt, st.session_state.current_topic_context)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("Please select a chapter and topic from the sidebar to view content and chat.")


