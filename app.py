import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Educational Assistant", page_icon="📚", layout="wide")
st.title('Educational Assistant')
st.header('Enhanced PDF Processor')
st.sidebar.title('Upload your PDFs here')

# Multiple file upload
user_file_uploads = st.sidebar.file_uploader(label='', type='pdf', accept_multiple_files=True)

def extract_subheadings(text):
    """Extract subheadings from the text using a heuristic approach."""
    return re.findall(r'(?m)^(?:[A-Z][^\n]{3,}|[0-9.]+\s+[^\n]+)$', text)

def split_text(text, max_tokens=1000):
    """Split text into smaller chunks based on the token limit."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(" ".join(current_chunk)) + len(word) + 1 > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
        current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def generate_output(chain, content):
    """Process content in chunks and combine the output."""
    text_chunks = split_text(content, max_tokens=1000)
    output = []
    for chunk in text_chunks:
        response = chain.invoke({'data': chunk})
        output.append(response)
    return "\n".join(output)

# Main content area
if user_file_uploads:
    # Dictionary to store data for each file
    pdf_data_dict = {}

    for user_file_upload in user_file_uploads:
        file_size = user_file_upload.size
        if file_size > 200 * 1024 * 1024:  # 200MB
            st.error(f"File {user_file_upload.name} exceeds 200MB. Please upload a smaller file.")
            continue

        # Read the uploaded file
        pdf_data = user_file_upload.read()

        # Save the uploaded file to a temporary location
        temp_file_path = f"temp_{user_file_upload.name}"
        with open(temp_file_path, "wb") as f:
            f.write(pdf_data)

        # Load the temporary PDF file
        loader = PyPDFLoader(temp_file_path)
        data = loader.load_and_split()
        os.remove(temp_file_path)  # Cleanup

        # Concatenate all data from the PDF
        full_text = "\n".join([page.page_content for page in data])

        # Extract subheadings from the PDF
        subheadings = extract_subheadings(full_text)

        # Store text chunks and subheadings in a dictionary
        pdf_data_dict[user_file_upload.name] = {
            "full_text": full_text,
            "subheadings": subheadings,
            "subheading_content": {subheading: "" for subheading in subheadings},
        }

        # Populate subheading content
        for subheading in subheadings:
            subheading_start = full_text.find(subheading)
            next_subheading_start = min(
                [full_text.find(next_heading) for next_heading in subheadings if full_text.find(next_heading) > subheading_start] or [len(full_text)]
            )
            pdf_data_dict[user_file_upload.name]["subheading_content"][subheading] = full_text[subheading_start:next_subheading_start]

    # Prompt templates
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a smart assistant. Provide a concise summary of the user's PDF content."),
        ("user", "{data}")
    ])

    quiz_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a smart assistant. Generate 10 quiz questions with 4 options from the user's PDF. Do not include answers."),
        ("user", "{data}")
    ])

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a smart assistant. Provide the correct answers and detailed explanations for each question."),
        ("user", "{data}")
    ])

    # Instantiate LLM and output parser
    llm = ChatGroq(model="llama3-70b-8192")
    output_parser = StrOutputParser()

    # Create chains
    summary_chain = summary_prompt | llm | output_parser
    quiz_chain = quiz_prompt | llm | output_parser
    answer_chain = answer_prompt | llm | output_parser

    # File dropdown selection in the sidebar
    selected_file = st.sidebar.selectbox("Select a file to process:", options=pdf_data_dict.keys())
    if selected_file:
        subheadings = pdf_data_dict[selected_file]["subheadings"]
        selected_subheading = st.sidebar.selectbox("Select a subheading:", options=subheadings)

        # Display subheading content in a collapsible section
        with st.expander(f"Content for: {selected_subheading}"):
            st.write(pdf_data_dict[selected_file]["subheading_content"][selected_subheading])

        # Create columns for quiz and answers
        col1, col2 = st.columns(2)

        # Generate quiz for selected subheading
        with col1:
            if st.button(f'Generate Quiz for "{selected_subheading}"'):
                with st.spinner(f'Generating quiz for "{selected_subheading}"...'):
                    content = pdf_data_dict[selected_file]["subheading_content"][selected_subheading]
                    quiz_questions = generate_output(quiz_chain, content)  # Process in chunks
                    st.write(f"### Quiz Questions for {selected_subheading}\n{quiz_questions}")

        # Generate answers and explanations
        with col2:
            if st.button(f'Generate Answers for "{selected_subheading}"'):
                with st.spinner(f'Generating answers for "{selected_subheading}"...'):
                    content = pdf_data_dict[selected_file]["subheading_content"][selected_subheading]
                    quiz_questions = generate_output(quiz_chain, content)  # Process questions
                    answers = generate_output(answer_chain, quiz_questions)  # Process answers
                    st.write(f"### Answers and Explanations for {selected_subheading}\n{answers}")

    # Button for summarizing all PDFs
    if st.button("Summarize All PDFs"):
        combined_summary = []
        for file_name, file_data in pdf_data_dict.items():
            with st.spinner(f'Generating summary for {file_name}...'):
                summary = generate_output(summary_chain, file_data["full_text"])  # Process in chunks
                combined_summary.append(f"### Summary for {file_name}\n{summary}")
        st.write("\n\n".join(combined_summary))

    # Button for detailed quiz output
    if st.button("Detailed Quiz Output"):
        for file_name, file_data in pdf_data_dict.items():
            st.write(f"### Quiz Questions and Explanations for {file_name}")
            for subheading, content in file_data["subheading_content"].items():
                st.write(f"#### Subheading: {subheading}")
                with st.spinner(f'Generating quiz and answers for "{subheading}"...'):
                    quiz_questions = generate_output(quiz_chain, content)
                    answers = generate_output(answer_chain, quiz_questions)
                    questions = quiz_questions.split("\n\n")
                    explanations = answers.split("\n\n")
                    for i, (question, explanation) in enumerate(zip(questions, explanations), start=1):
                        st.write(f"*Question {i}: {question}\n\nExplanation*:\n{explanation}\n")