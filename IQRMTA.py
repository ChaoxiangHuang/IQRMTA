# Corrected placement for set_page_config
import streamlit as st
st.set_page_config(layout="wide")

import json
import os
import re
from openai import OpenAI

# --- Configuration ---
JSON_FILE = "iqrm_chapter_summaries.json"
API_KEY = "sk-e5UCdVT-knC7iEtrKwccstG3yZrf_i-hrKJ4dv-QpsT3BlbkFJr8PMyVYX7MJ7XipoyCL8HbAtYrbikPRvaioYVwkakA" # IMPORTANT: Use Streamlit secrets for API keys in deployed apps!

# --- OpenAI Client ---
try:
    client = OpenAI(api_key=API_KEY)
except Exception as e:
    st.warning(f"Failed to initialize OpenAI client: {e}. Q&A feature will be disabled.")
    client = None

# --- Load Knowledge Base ---
@st.cache_data
def load_knowledge_base(file_path):
    if not os.path.exists(file_path):
        return None, f"Error: Knowledge base file not found at {file_path}"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            sorted_chapter_keys = sorted(
                [key for key in data if key.startswith('chapter_') and key.split('_')[1].isdigit()],
                key=lambda x: int(x.split('_')[1])
            )
            return {ch_key: data[ch_key] for ch_key in sorted_chapter_keys}, None
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

# --- URL Parameter Handling with Pinpoint Debugging ---
def get_topic_from_url(kb):
    # These lines are from your confirmed successful debug output
    all_info_values = st.query_params.get_all("info")
    first_info_value = st.query_params.get("info") # Your debug showed this is "chapter1_Array_Calculations"

    # --- Display confirmed values ---
    st.sidebar.markdown("--- DEBUG (get_topic_from_url) ---")
    st.sidebar.write("`st.query_params.get_all('info')`:")
    st.sidebar.json(all_info_values)
    st.sidebar.write(f"`st.query_params.get('info')` (this will be used as `param_to_process`): `{first_info_value}`")
    # --- End Display ---

    param_to_process = first_info_value # Use the variable we just confirmed

    target_chapter = None
    target_topic   = None
    error_message  = None

    if param_to_process:
        # --- PINPOINT DEBUG BEFORE REGEX ---
        st.sidebar.write(f"**Just before `re.match`, `param_to_process` is:** `{param_to_process}` (Type: {type(param_to_process).__name__})")
        print(f"TERMINAL DEBUG - Just before `re.match`, `param_to_process` is: '{param_to_process}'")
        # --- END PINPOINT DEBUG ---

        m = re.match(r"chapter(\d+)[-_](.+)", param_to_process, re.IGNORECASE)
        if m:
            st.sidebar.write("✅ Regex Matched!")
            print("TERMINAL DEBUG - Regex Matched!")
            chap_no   = m.group(1)
            url_topic_segment = m.group(2)
            slug_for_matching = url_topic_segment.replace("_", " ").replace("-", " ")
            chap_key  = f"chapter_{chap_no}"

            if kb and chap_key in kb:
                for t_key_in_kb in kb[chap_key]:
                    if t_key_in_kb.lower() == slug_for_matching.lower():
                        target_chapter = chap_key
                        target_topic   = t_key_in_kb
                        break
                if not target_topic:
                    error_message = f"Topic '{slug_for_matching}' (from URL '{url_topic_segment}') not found in Chapter {chap_no}."
            else:
                error_message = f"Chapter {chap_no} (key '{chap_key}') not found."
        else:
            # --- PINPOINT DEBUG INSIDE REGEX FAILURE (ELSE BLOCK) ---
            st.sidebar.write(f"❌ Regex FAILED. **Inside `else` block, `param_to_process` is:** `{param_to_process}`")
            print(f"TERMINAL DEBUG - Regex FAILED. Inside `else` block, `param_to_process` is: '{param_to_process}'")
            # --- END PINPOINT DEBUG ---
            error_message = (
                f"Invalid format: '{param_to_process}'. " # Error uses the same variable
                "Use chapter<Number>-<TopicName> or chapter<Number>_<TopicName>."
            )
    else:
        st.sidebar.write("`param_to_process` was None or empty (no 'info' QParam).")
        print("TERMINAL DEBUG - `param_to_process` was None or empty.")


    st.sidebar.write(f"Returning: chapter=`{target_chapter}`, topic=`{target_topic}`, error=`{error_message}`")
    st.sidebar.markdown("--- END DEBUG ---")
    print(f"TERMINAL DEBUG - Returning: chapter={target_chapter}, topic={target_topic}, error={error_message}")
    return target_chapter, target_topic, error_message

# --- Initialize Session State (Good as is) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_topic_context" not in st.session_state:
    st.session_state.current_topic_context = ""
# These will be set by widgets or URL logic, but initializing to None is fine
if "selected_chapter_key" not in st.session_state:
    st.session_state.selected_chapter_key = None
if "selected_topic_name" not in st.session_state:
    st.session_state.selected_topic_name = None
if "_previous_topic" not in st.session_state:
    st.session_state._previous_topic = None


# --- Main App Logic ---
st.title("IQRM AI Teaching Assistant")

if load_error:
    st.error(load_error)
    st.stop()

if not knowledge_base:
    st.warning("Knowledge base is empty or could not be loaded.")
    st.stop()

# Get target chapter/topic from URL
url_target_chapter_key, url_target_topic_name, url_parse_error = get_topic_from_url(knowledge_base)

if url_parse_error:
    st.warning(url_parse_error) # Display the parsing error, e.g., "Invalid 'info' parameter format: 'c'"

# --- Determine Initial Selections for Widgets ---
chapter_options_keys = list(knowledge_base.keys())
chapter_options_display = [f"Chapter {key.split('_')[1]}" for key in chapter_options_keys]

# Determine initial chapter
# Priority: 1. Valid URL param, 2. Valid Session State, 3. First chapter
initial_chapter_key_to_select = None
initial_chapter_index_to_select = 0 # Default to first chapter

if url_target_chapter_key and url_target_chapter_key in chapter_options_keys:
    initial_chapter_key_to_select = url_target_chapter_key
elif st.session_state.selected_chapter_key and st.session_state.selected_chapter_key in chapter_options_keys:
    initial_chapter_key_to_select = st.session_state.selected_chapter_key
elif chapter_options_keys: # If no URL/session, but chapters exist
    initial_chapter_key_to_select = chapter_options_keys[0]

if initial_chapter_key_to_select:
    initial_chapter_index_to_select = chapter_options_keys.index(initial_chapter_key_to_select)
else: # Should not happen if knowledge_base has chapters, but as a fallback
    st.warning("No chapters available for selection.")
    st.stop()


# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")

# Chapter Selectbox
# The key 'chapter_select_widget' will store the selected index in st.session_state.chapter_select_widget
selected_chapter_idx = st.sidebar.selectbox(
    "Select Chapter:",
    range(len(chapter_options_keys)),
    index=initial_chapter_index_to_select,
    format_func=lambda i: chapter_options_display[i],
    key="chapter_select_widget" # Widget key
)
selected_chapter_key = chapter_options_keys[selected_chapter_idx]
st.session_state.selected_chapter_key = selected_chapter_key # Store the actual key string for logic

# Topic Selectbox
current_chapter_data = knowledge_base.get(selected_chapter_key, {})
topic_options_keys = list(current_chapter_data.keys())
selected_topic_name = None # Initialize

if not topic_options_keys:
    st.sidebar.warning(f"No topics found for Chapter {selected_chapter_key.split('_')[1]}.")
else:
    initial_topic_name_to_select = None
    initial_topic_index_to_select = 0 # Default to first topic

    # Determine initial topic FOR THE CURRENTLY SELECTED CHAPTER
    # Priority: 1. Valid URL param (if chapter matches), 2. Valid Session State (if chapter matches), 3. First topic
    if url_target_topic_name and selected_chapter_key == url_target_chapter_key and url_target_topic_name in topic_options_keys:
        initial_topic_name_to_select = url_target_topic_name
    elif st.session_state.selected_topic_name and \
         st.session_state.get("selected_chapter_key_for_topic") == selected_chapter_key and \
         st.session_state.selected_topic_name in topic_options_keys:
        initial_topic_name_to_select = st.session_state.selected_topic_name
    elif topic_options_keys: # If no URL/session match for this chapter, but topics exist
        initial_topic_name_to_select = topic_options_keys[0]

    if initial_topic_name_to_select:
        initial_topic_index_to_select = topic_options_keys.index(initial_topic_name_to_select)

    selected_topic_idx = st.sidebar.selectbox(
        "Select Topic:",
        range(len(topic_options_keys)),
        index=initial_topic_index_to_select,
        format_func=lambda i: topic_options_keys[i], # Display actual topic name
        key="topic_select_widget" # Widget key
    )
    selected_topic_name = topic_options_keys[selected_topic_idx]
    st.session_state.selected_topic_name = selected_topic_name # Store the actual name string
    st.session_state.selected_chapter_key_for_topic = selected_chapter_key # Remember which chapter this topic belongs to


# --- Main Area ---
if selected_chapter_key and selected_topic_name:
    st.header(f"Chapter {selected_chapter_key.split('_')[1]}: {selected_topic_name}")

    topic_data = knowledge_base[selected_chapter_key][selected_topic_name]

    new_context = f"Topic: {selected_topic_name}\nSummary: {topic_data.get('summary', '')}\nSubtitles: {', '.join(topic_data.get('subtitles', []))}"
    if st.session_state.current_topic_context != new_context or st.session_state._previous_topic != selected_topic_name:
        st.session_state.current_topic_context = new_context
        st.session_state.messages = []
        st.session_state._previous_topic = selected_topic_name
        st.rerun()

    # Expand details if this specific topic was loaded via URL
    # This logic should ideally only make it expanded on the *very first* load from URL.
    # A simple way is to check if the selectboxes' current values match the URL targets.
    expand_details_default = False
    if url_target_chapter_key == selected_chapter_key and \
       url_target_topic_name == selected_topic_name and \
       not st.session_state.get("_url_params_processed", False): # Check a flag
        expand_details_default = True
    if selected_chapter_key or selected_topic_name: # After any selection, mark as processed
        st.session_state._url_params_processed = True


    with st.expander("View Topic Details (Summary, Subtitles, Quiz, Tasks)", expanded=expand_details_default):
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

    st.divider()
    st.subheader(f"Chat about: {selected_topic_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask a question about {selected_topic_name}..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = get_ai_response(prompt, st.session_state.current_topic_context)
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.info("Please select a chapter and topic from the sidebar to view content and chat.")

# To reset the URL processed flag if user navigates away and back via URL (optional, advanced)
# This is tricky as reruns happen often. A simple flag might not be enough for complex state.
# if not (url_target_chapter_key == selected_chapter_key and url_target_topic_name == selected_topic_name):
#    if "_url_params_processed" in st.session_state:
#        del st.session_state._url_params_processed
