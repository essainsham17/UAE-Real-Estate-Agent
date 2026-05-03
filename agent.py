import pandas as pd
import numpy as np
from typing import TypedDict
import os
from langchain_groq import ChatGroq
import joblib
from langgraph.graph import StateGraph, START,END
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model='llama-3.3-70b-versatile', temperature=0)

class AgentState(TypedDict):
    user_query:str
    Location:str
    Bedrooms:int
    predicted_price:float
    final_response:str

    down_payment:float
    monthly_payment:float


def extract_property_info(state: AgentState):
    text=state["user_query"]
    prompt=f"""
    you are a data extraction bot.
    extract the location and the number of bedrooms from this user query:"{text}".
    Rules:
    1. return only comma seperated string
    2. format: Location, Bedrooms
    3. do NOT add any conversational text
    4. Example: JVC, 2
    """

    response=llm.invoke(prompt)
    extracted_content=response.content

    try:
        parts=extracted_content.split(',')
        location=parts[0].strip()
        n_bedrooms=int(parts[1].strip())
    except Exception as e:
        location= "Unknown"
        n_bedrooms=0

    return({'Location':location,'Bedrooms':n_bedrooms})
def Predictor(state: AgentState):
    location=state["Location"]
    n_bedrooms=state["Bedrooms"]

    prediction_model=joblib.load("uae_real_estate_model.pkl")
    columns=joblib.load("model_columns.pkl")

    df=pd.DataFrame({col: 0 for col in columns},index=[0])
    if location in columns:
        df[location]=1
    df['Area(Sqft)']=(n_bedrooms*750)+24
    df['Bedrooms']=n_bedrooms
    
    result=prediction_model.predict(df)
    final_result=round(result[0],2)

    return {'predicted_price':final_result}
def Mortgage_calculator(state: AgentState):
    price=state['predicted_price']
    loan=price*0.8
    interest=0.045/12
    months=25*12
    
    EMI=round(loan*(interest*(interest+1)**months)/((interest+1)**months-1),2)

    return {"down_payment":(price*0.2),"monthly_payment":EMI}
def Generate_Response(state: AgentState):
    query=state['user_query']
    location=state['Location']
    bedrooms=state['Bedrooms']
    price=state['predicted_price']
    down_payment=state['down_payment']
    monthly_payment=state['monthly_payment']

    prompt=f'''
    you are a UAE real estate AI. generate a professional response to the users query:{query}.
    Compare the users expected price in the query with the actual model predicted price = {price}.
    also consider the location {location}.
    Also give the client about the down payment of the property is {down_payment} which is 20% of the total price.
    Monthly payment for 25 years with 4.5% interest rate is calculated at {monthly_payment}'''

    response=llm.invoke(prompt)

    return {'final_response':response.content}

workflow=StateGraph(AgentState)

workflow.add_node("extractor",extract_property_info)
workflow.add_node("perdictor",Predictor)
workflow.add_node("responder",Generate_Response)
workflow.add_node('Mortgage',Mortgage_calculator)

workflow.add_edge(START,"extractor")
workflow.add_edge("extractor","perdictor")
workflow.add_edge("perdictor","Mortgage")
workflow.add_edge("Mortgage","responder")
workflow.add_edge('Mortgage',END)

real_estate_agent=workflow.compile()

