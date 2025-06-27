# Corrected placement for set_page_config
import streamlit as st
st.set_page_config(layout="wide")

import json
import os
import re
from openai import OpenAI

# --- Configuration ---
### NOTE: Make sure this filename matches your JSON file.
JSON_FILE = "iqrm_summaries_answers.json"

API_KEY = "sk-e5UCdVT-knC7iEtrKwccstG3yZrf_i-hrKJ4dv-QpsT3BlbkFJr8PMyVYX7MJ7XipoyCL8HbAtYrbikPRvaioYVwkakA"

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
            # Sort chapters numerically
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
    if not client or "YOUR_API_KEY" in API_KEY:
        return "Error: OpenAI client not initialized or API key is invalid."
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
    # Get the value of the 'info' query parameter
    param_value = st.query_params.get("info")
    target_chapter = None
    target_topic   = None
    error_message  = None

    if param_value:
        # Regex to capture chapter number and topic name
        m = re.match(r"chapter(\d+)[-_](.+)", param_value, re.IGNORECASE)
        if m:
            chap_no   = m.group(1)
            url_topic_slug = m.group(2).replace("_", " ").replace("-", " ")
            chap_key  = f"chapter_{chap_no}"

            if kb and chap_key in kb:
                # Find the topic key by case-insensitive matching
                for t_key_in_kb in kb[chap_key]:
                    if t_key_in_kb.lower() == url_topic_slug.lower():
                        target_chapter = chap_key
                        target_topic   = t_key_in_kb
                        break
                if not target_topic:
                    error_message = f"Topic '{url_topic_slug}' not found in Chapter {chap_no}."
            else:
                error_message = f"Chapter {chap_no} (key '{chap_key}') not found."
        else:
            error_message = f"Invalid format for 'info' parameter: '{param_value}'. Use 'chapter<Number>-<TopicName>'."
    
    return target_chapter, target_topic, error_message

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_topic_context" not in st.session_state:
    st.session_state.current_topic_context = ""
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

# --- URL and Widget Selection Logic ---
url_target_chapter_key, url_target_topic_name, url_parse_error = get_topic_from_url(knowledge_base)
if url_parse_error:
    st.warning(url_parse_error)

chapter_options_keys = list(knowledge_base.keys())
chapter_options_display = [f"Chapter {key.split('_')[1]}" for key in chapter_options_keys]

# Determine initial chapter selection
initial_chapter_key_to_select = chapter_options_keys[0] if chapter_options_keys else None
if url_target_chapter_key and url_target_chapter_key in chapter_options_keys:
    initial_chapter_key_to_select = url_target_chapter_key
elif st.session_state.selected_chapter_key and st.session_state.selected_chapter_key in chapter_options_keys:
    initial_chapter_key_to_select = st.session_state.selected_chapter_key

initial_chapter_index_to_select = chapter_options_keys.index(initial_chapter_key_to_select) if initial_chapter_key_to_select else 0

st.sidebar.title("Navigation")
selected_chapter_idx = st.sidebar.selectbox(
    "Select Chapter:",
    range(len(chapter_options_keys)),
    index=initial_chapter_index_to_select,
    format_func=lambda i: chapter_options_display[i],
    key="chapter_select_widget"
)
selected_chapter_key = chapter_options_keys[selected_chapter_idx]

# Update session state for the chapter
if st.session_state.selected_chapter_key != selected_chapter_key:
    st.session_state.selected_chapter_key = selected_chapter_key
    # When chapter changes, reset the topic to avoid state inconsistencies
    st.session_state.selected_topic_name = None 
    st.rerun()

current_chapter_data = knowledge_base.get(selected_chapter_key, {})
topic_options_keys = list(current_chapter_data.keys())
selected_topic_name = None

if topic_options_keys:
    # Determine initial topic selection
    initial_topic_name_to_select = topic_options_keys[0]
    if url_target_topic_name and selected_chapter_key == url_target_chapter_key and url_target_topic_name in topic_options_keys:
        initial_topic_name_to_select = url_target_topic_name
    elif st.session_state.selected_topic_name and st.session_state.selected_topic_name in topic_options_keys:
        initial_topic_name_to_select = st.session_state.selected_topic_name

    initial_topic_index_to_select = topic_options_keys.index(initial_topic_name_to_select)

    selected_topic_idx = st.sidebar.selectbox(
        "Select Topic:",
        range(len(topic_options_keys)),
        index=initial_topic_index_to_select,
        format_func=lambda i: topic_options_keys[i],
        key="topic_select_widget"
    )
    selected_topic_name = topic_options_keys[selected_topic_idx]
    st.session_state.selected_topic_name = selected_topic_name

# --- Main Area Display ---
if selected_chapter_key and selected_topic_name:
    st.header(f"Chapter {selected_chapter_key.split('_')[1]}: {selected_topic_name}")

    topic_data = knowledge_base[selected_chapter_key][selected_topic_name]

    # Context management for the chatbot
    new_context = f"Topic: {selected_topic_name}\nSummary: {topic_data.get('summary', '')}"
    if st.session_state.current_topic_context != new_context:
        st.session_state.current_topic_context = new_context
        st.session_state.messages = [] # Reset chat history when topic changes
        # Use a flag to avoid rerunning if it's the first load
        if st.session_state.get('_previous_topic') is not None:
             st.rerun()
    st.session_state._previous_topic = selected_topic_name

    # Logic to expand details on first URL load
    expand_details_default = False
    if url_target_chapter_key == selected_chapter_key and url_target_topic_name == selected_topic_name and not st.session_state.get("_url_params_processed", False):
        expand_details_default = True
    if selected_chapter_key or selected_topic_name:
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

        ### MODIFIED SECTION: Interactive Quiz ###
        st.subheader("Interactive Quiz")
        quiz_questions = topic_data.get("quiz_questions", [])

        if quiz_questions and isinstance(quiz_questions[0], dict): # Check if data is in the new format
            # Use a form to collect all answers before submitting
            form_key = f"quiz_form_{selected_chapter_key}_{selected_topic_name}"
            with st.form(key=form_key):
                user_answers = {}
                for i, q_data in enumerate(quiz_questions):
                    question = q_data.get("question")
                    # The choices are a dictionary {'A': 'text', ...}, convert to a list for the radio widget
                    choices_dict = q_data.get("choices", {})
                    # Format options like "A: Answer text" to be displayed
                    formatted_options = [f"{key}: {value}" for key, value in choices_dict.items()]
                    
                    if question and formatted_options:
                        # Use a unique key for each radio widget
                        radio_key = f"q_{i}"
                        user_answers[radio_key] = st.radio(
                            f"**{i+1}. {question}**",
                            options=formatted_options,
                            index=None,  # Default to no selection
                            key=f"radio_{form_key}_{i}" # Make keys absolutely unique
                        )
                
                submitted = st.form_submit_button("Check Answers")

            if submitted:
                score = 0
                total_questions = len(quiz_questions)
                
                st.write("---")
                st.subheader("Quiz Results")

                for i, q_data in enumerate(quiz_questions):
                    radio_key = f"q_{i}"
                    user_answer_full = user_answers.get(radio_key) # e.g., "B: Some text" or None
                    user_answer_choice = user_answer_full.split(':')[0] if user_answer_full else None # "B"
                    
                    correct_choice_key = q_data.get("correct") # "B"
                    correct_answer_text = q_data.get("choices", {}).get(correct_choice_key, "")

                    is_correct = (user_answer_choice == correct_choice_key)

                    if is_correct:
                        score += 1
                        st.success(f"**Question {i+1}: Correct!**")
                    else:
                        st.error(f"**Question {i+1}: Incorrect.**")

                    st.write(f"> {q_data.get('question')}")
                    st.write(f"Your answer: `{user_answer_full if user_answer_full else 'No answer selected'}`")
                    if not is_correct:
                        st.write(f"Correct answer: `{correct_choice_key}: {correct_answer_text}`")
                    st.write("---")

                # Display final score
                final_score_text = f"Your final score: {score} out of {total_questions}"
                if score == total_questions:
                    st.balloons()
                    st.success(final_score_text + " - Perfect! Great job!")
                else:
                    st.info(final_score_text)
        else:
            st.write("No interactive quiz questions available for this topic.")
        ### END OF MODIFIED SECTION ###


        st.subheader("Training Tasks")
        training_tasks = topic_data.get("training_tasks", [])
        if training_tasks:
            for i, t in enumerate(training_tasks):
                st.markdown(f"{i+1}. {t}")
        else:
            st.write("No training tasks available.")

    # --- Chatbot Interface ---
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

