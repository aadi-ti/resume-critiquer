import streamlit as st
import PyPDF2
import os
import io
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

st.set_page_config(page_title="Resume Critiquer", page_icon=":page_with_curl:", layout="centered")
st.title("Resume Critiquer :page_with_curl:")
st.markdown("Upload your resume to get AI-powered feedback tailored to your needs!")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

job_role = st.text_input(
    "Enter the job role you are applying for:",
    placeholder="e.g., Software Engineer, Data Scientist"
)

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

analyze_resume = st.button("Analyze Resume")
if analyze_resume:
    if not uploaded_file:
        st.error("Please upload a resume file.")
    elif not job_role:
        st.error("Please enter the job role you are applying for.")
    else:
        try:
            file_content = extract_text_from_file(uploaded_file)
            if not file_content.strip():
                st.error("The uploaded file has no readable content. Please check the file and try again.")
                st.stop()

            prompt = f"""You are an expert resume reviewer with years of experience in HR and recruitment.
Please analyze this resume and provide constructive feedback.
Focus on the following aspects:
1. content clarity and impact
2. skill presentation
3. experience description
4. specific improvements for {job_role if job_role else 'general job applications'}

Resume Content: {file_content}
Please provide your feedback in a concise and actionable format.
Please ensure that the feedback is relevant to the job role provided.
Please provide your analysis in a clear and structured format, with specific recommendations for improvement.
also give a score out of 10 for the overall quality of the resume.
also provide a summary of the key strengths and weaknesses of the resume.
also provide a list of keywords that are relevant to the job role provided.
also provide a list of common mistakes to avoid in resumes for the job role provided.
also provide a list of resources for further improvement.
also provide a list of common interview questions for the job role provided.
also provide an example of changes to be made to the resume based on the analysis.
also maintain spacing and formatting in the response.
"""
            client = InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=huggingface_api_key)
            response = client.text_generation(
                prompt,
                max_new_tokens=1000,
                temperature=0.7
            )
            st.success("Resume and job role submitted successfully! Here is your analysis:")
            st.markdown("### Analysis Results:")
            st.markdown(response)
        except Exception as e:
            st.error(f"An error occurred while processing the resume: {e}")