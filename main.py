from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
import streamlit as st
import os

# Set up API key for Google's Generative AI
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

# Create a prompt template for generating recommendations
recommendation_template = "Give me the top 5 recommendations to visit in {country} for {adults} adults and {kids} kids traveling on {date} with activity plans"

recommendation_prompt = PromptTemplate(template=recommendation_template, input_variables=['country', 'adults', 'kids', 'date'])

# Initialize Google's Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

# Create the LLM chain for generating recommendations
recommendation_chain = recommendation_prompt | gemini_model

# Streamlit app setup
st.header("Top 5 Travel Recommendations")

st.subheader("Discover the must-visit places")

countries = ["India", "Netherlands", "Paris", "Thailand", "Germany", "Dubai", "Russia", "Portugal", "Spain", "Hungary"]
country = st.selectbox("Select a country:", options=countries)

date = st.date_input("Date of Travel:")
adults = st.number_input("Number of Adults:", min_value=1, step=1, value=1)
kids = st.number_input("Number of Kids:", min_value=0, step=1, value=0)

if st.button("Generate Recommendations"):
    if country.strip():
        recommendations = recommendation_chain.invoke({"country": country, "adults": adults, "kids": kids, "date": date})
        st.write(recommendations.content)
    else:
        st.warning("Please select a valid country.")
