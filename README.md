---
title: UAE Real Estate Agent
emoji: 🏢
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---


# 🏘️ Intelligent UAE Real Estate Advisory Agent

An end-to-end, AI-powered real estate advisor designed to evaluate property prices and calculate mortgage estimates in the United Arab Emirates (Dubai & Abu Dhabi). 

This application utilizes an agentic workflow to combine predictive machine learning models with deterministic financial calculations and Large Language Model (LLM) orchestration.

## 🧠 Architecture & Tech Stack

This project is built using a multi-node, state-managed AI architecture:
*   **Frontend:** Streamlit (Conversational UI with session state management)
*   **Orchestration:** LangGraph (StateGraph for multi-step agent routing)
*   **LLM Engine:** Groq API (High-speed inference)
*   **Machine Learning:** Scikit-Learn (RandomForestRegressor for predictive market pricing)
*   **Environment:** Conda & Python-dotenv (Secure dependency and secrets management)

## ⚙️ How It Works

The agent relies on a structured `AgentState` dictionary to pass context between specialized nodes:
1.  **Extraction Node:** The LLM analyzes the user's natural language query to extract key parameters (Location, Size, Price).
2.  **Prediction Node:** A pre-trained Random Forest model (`joblib`) ingests the extracted parameters to predict the fair market value of the property based on historical UAE market data.
3.  **Calculation Node:** A deterministic Python function calculates estimated monthly mortgage payments based on the property price and current UAE interest rates.
4.  **Synthesis Node:** The LLM takes the user's original query, the ML prediction, and the math calculation to generate a highly contextual, professional, and accurate response.

## 🚀 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/Intelligent-UAE-Real-Estate-Advisory-Agent.git](https://github.com/yourusername/Intelligent-UAE-Real-Estate-Advisory-Agent.git)
   cd Intelligent-UAE-Real-Estate-Advisory-Agent