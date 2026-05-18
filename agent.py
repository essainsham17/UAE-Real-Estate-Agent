import pandas as pd
import numpy as np
from typing import TypedDict
import os
from langchain_groq import ChatGroq
import joblib
from langgraph.graph import StateGraph, START,END
from dotenv import load_dotenv
from pydantic import BaseModel , Field

load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

llm=ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
prediction_model=joblib.load("price_prediction_model.pkl")
columns=joblib.load("model_columns.pkl")

class PropertyExtraction(BaseModel):
    Beds: int = Field(
        default=1, description="the number of bedrooms. set the default to 1 if user did'nt specify"
    )
    Baths: int = Field(
        description="the number of bathrooms. if not specified assume bedrooms+1"
    )
    Type: str =Field(
        default='Apartment',description="the type of property (eg: Villa,Apartment). default set to Apartment if not specified"
    )
    Area_in_sqft: float = Field(
        description="Total area in square feet. If not explicitly mentioned, estimate it by multiplying the number of Bedrooms by 750, plus 24."
    )
    Furnishing: str = Field(
        default="Unfurnished", 
        description="Furnishing status (Furnished or Unfurnished). Default to 'Unfurnished'."
    )
    Location: str = Field(
        description="the exact location of the property. if not mensioned return 'UNKNOWN'"
    )
    City: str = Field(
        default="Dubai", 
        description="The city. Default to 'Dubai'."
    )

class AgentState(TypedDict):
    user_query:str

    
    Beds: int
    Baths: int
    Type: str
    Area_in_sqft: float
    Furnishing: str
    Location:str
    City: str

    
    predicted_price:float
    final_response:str
    down_payment:float
    monthly_payment:float


def extract_property_info(state: AgentState):
    text=state["user_query"]
    
    structured_llm=llm.with_structured_output(PropertyExtraction)
    extracted_content=structured_llm.invoke(text)
    
    return (
        {
        'Beds':extracted_content.Beds,
        'Baths':extracted_content.Baths,
        'Type':extracted_content.Type,
        'Area_in_sqft':extracted_content.Area_in_sqft,
        'Furnishing':extracted_content.Furnishing,
        'Location':extracted_content.Location,
        'City':extracted_content.City
        }
    )

    
def Predictor(state: AgentState):
    input_data={
        'Beds':[state['Beds']],
        'Baths':[state['Baths']],
        'Type':[state['Type']],
        'Area_in_sqft':[state['Area_in_sqft']],
        'Furnishing':[state['Furnishing']],
        'Location':[state['Location']],
        'City':[state['City']]
    }
    
    df=pd.DataFrame(input_data)
    
    
    result=prediction_model.predict(df)
    final_result=np.exp(result[0])

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
    Beds=state['Beds']
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
workflow.add_node("predictor",Predictor)
workflow.add_node("responder",Generate_Response)
workflow.add_node('Mortgage',Mortgage_calculator)

workflow.add_edge(START,"extractor")
workflow.add_edge("extractor","predictor")
workflow.add_edge("predictor","Mortgage")
workflow.add_edge("Mortgage","responder")
workflow.add_edge('responder',END)

real_estate_agent=workflow.compile()


