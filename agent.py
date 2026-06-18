import pandas as pd
import numpy as np
from typing import TypedDict, Optional
import os
from langchain_groq import ChatGroq
import joblib
from langgraph.graph import StateGraph, START,END
from dotenv import load_dotenv
from pydantic import BaseModel , Field
from langgraph.checkpoint.memory import MemorySaver
from langchain_tavily import TavilySearch
import datetime
from datetime import date

memory=MemorySaver()
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"]=os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGHAIN_PROJECT")

search_tool=TavilySearch(
    max_results=5,
    api_key=os.getenv("TAVILY_API_KEY")
)

llm=ChatGroq(model='llama-3.3-70b-versatile', temperature=0)
prediction_model=joblib.load("price_prediction_model.pkl")
columns=joblib.load("model_columns.pkl")

class PropertyExtraction(BaseModel):
    Beds: Optional[int] = Field(
        default=None,
        description="the number of bedrooms. set the default to None if user did'nt specify"
    )
    Baths: int = Field(
        default=2, description="the number of bathrooms. if not specified assume bedrooms+1"
    )
    Type: Optional[str] =Field(
        default=None,
        description="the type of property (eg: Villa,Apartment). default set to None if not specified"
    )
    Area_in_sqft: float = Field(
        default=750, description="Total area in square feet. If not explicitly mentioned, estimate it by multiplying the number of Bedrooms by 750, plus 24."
    )
    Furnishing: Optional[str] = Field(
        default=None,
        description="Furnishing status (Furnished or Unfurnished). Default to None."
    )
    Location: Optional[str] = Field(
        default=None,
        description="the exact location of the property. if not mentioned return None"
    )
    City: str = Field(
        default="Dubai", 
        description="The city. Default to 'Dubai'."
    )

class AgentState(TypedDict):
    user_query:str
    validation_status: str
    user_intent:str

    
    Beds: int
    Baths: int
    Type: str
    Area_in_sqft: float
    Furnishing: str
    Location:str
    City: str

    
    predicted_price:int
    final_response:str
    monthly_rent:float
    annual_rent:float
    security_deposit: float
    agent_commission: float
    total_amount: float

def classify_intent(state: AgentState):
    query=state.get('user_query')
    classification_prompt = f"""Analyze the user query: '{query}'. 
    - If they are looking to rent/find a specific property, or providing details like bedrooms and location, reply 'property'.
    - If they are asking for market news, geopolitical impacts, trends, or general real estate information, reply 'research'.
    - If it's just a hello/greeting, reply 'greeting'.
    Reply with ONLY ONE word."""
    intent=llm.invoke(classification_prompt)

    if 'greeting' in intent.content.lower():
        prompt=f"You are a UAE real estate rental advisor. The user said: '{query}'. Respond warmly and let them know you can help them find rental properties in the UAE"
        res=llm.invoke(prompt)
        return {'user_intent':'greeting','final_response':res.content}
    elif 'property' in intent.content.lower():
        return {'user_intent':'property'}
    elif 'research' in intent.content.lower():
        return {'user_intent':'research'}
    else:
        result=llm.invoke(f"i could not classify the user intent for the query: '{query}'. Please respond with a warm, professional message to the user, letting them know you can help them find rental properties in the UAE.")
        return {'final_response':result.content}
    

def extract_property_info(state: AgentState):
    text=state["user_query"]
    Beds=state.get('Beds')
    Baths=state.get('Baths')
    Type=state.get('Type')
    Area_in_sqft=state.get('Area_in_sqft')
    Furnishing=state.get('Furnishing')
    Location=state.get('Location')
    City=state.get('City')

    context=f"""
    existing property details are: Beds={Beds}, Baths={Baths}, Type={Type}, Area_in_sqft={Area_in_sqft}, 
    Furnishing={Furnishing}, Location={Location}, City={City}
    new user message= {text}
    Only update a field if the new message explicitly mentions it. If a field already has a value keep it exactly as is. Never set a field back to None.
    """
    structured_llm=llm.with_structured_output(PropertyExtraction)
    extracted_content=structured_llm.invoke(context)
    
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

    
def validate_inputs(state: AgentState):
    Beds=state.get('Beds')
    Type=state.get('Type')
    Furnishing=state.get('Furnishing')
    Location=state.get('Location')

    missing_fields=[]
    if Beds==None:
        missing_fields.append('Number of Bedrooms')
    if Type==None:
        missing_fields.append('Property Type: Apartment, Townhouse or Villa')
    if Furnishing==None:
        missing_fields.append('furnishing Preference: Furnished or Unfurnished')
    if Location==None:
        missing_fields.append('preferred Location or neighbourhood')

    if missing_fields:
        missing_str=', '.join(missing_fields)
        prompt=f""" You are a professional UAE real estate advisor. Ask the client the following in a warm, natural, conversational way: {missing_str}. Keep it to one sentence."""
        question=llm.invoke(prompt).content
        return {'validation_status':'Incomplete', 'final_response':question}
    else:
        return {'validation_status':'Complete'}


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
    final_result=float(np.exp(result[0]))

    return {'predicted_price':int(final_result)}


def Rental_calculator(state: AgentState):
    Annual_rent=state['predicted_price']
    
    Monthly_rent=Annual_rent/12
    security_deposit=Annual_rent*0.05
    agent_commission=Annual_rent*0.02
    total_amount=Monthly_rent+security_deposit+agent_commission
    
    return {"monthly_rent":Monthly_rent,'annual_rent':Annual_rent, 'security_deposit':security_deposit, 'agent_commission': agent_commission,'total_amount':total_amount}



def Generate_Response(state: AgentState):
    query=state['user_query']
    location=state['Location']
    price=state['predicted_price']
    monthly_rent=state['monthly_rent']
    annual_rent=state['annual_rent']
    security_deposit=state['security_deposit']
    agent_commission=state['agent_commission']
    total_amount=state.get('total_amount')

    name="ESSA INSHAM"
    num="+971 52 718 1331"
    linkedin="https://www.linkedin.com/in/essa-insham"

    prompt=f""" you are a uae real estate rental advisor.respond in warm, helpfull and professional tone to  generate a professional response to the users query: {query}.
    compare the users expected price in the query with the actual model predicted price = {price}.
    also consider the location: {location}.
    also give the client information about monthly rent {monthly_rent}, annual rent:{annual_rent}, security_deposit of 5% of annual rent is: {security_deposit}
and agent commission is 2% of annual rent is {agent_commission}.
Also inform the client that the total upfront cost including security deposit and agent commission is {total_amount}
also add my name:{name} , my mobile number: {num} and my linkedin:{linkedin}.
    """

    response=llm.invoke(prompt)

    return {'final_response':response.content}
def research(state: AgentState):
    query=state['user_query']
    search_response=search_tool.invoke({"query":query})
    results=search_response.get('results', []) if isinstance(search_response, dict) else search_response
    if not results:
        return {'final_response':"No results found for your query. Try rephrasing your question or ask about a different topic."}
    combined_context = "\n\n".join(
        f"Source {i+1} ({r.get('title', 'Untitled')}): {r.get('content', '')}"
        for i, r in enumerate(results)
    )
    today_str = date.today().strftime("%B %d, %Y")

    prompt= f"""You are a UAE real estate market advisor. Today's date is {today_str}.

A client asked: '{query}'

Here is recent information gathered from multiple sources:
{combined_context}

Using only the information above, write a clear, well-rounded answer that directly addresses the client's question.
Start your response by referencing today's date, e.g. "As of {today_str}, ...".
Synthesize the sources into a coherent response rather than quoting any single one. If sources disagree, mention both
perspectives. Keep it professional and concise (3-5 sentences)."""
    response=llm.invoke(prompt)
    return {'final_response':response.content}

def rout_validator(state: AgentState):
    if state.get('validation_status').lower() == 'incomplete':
        return(END)
    if state.get('validation_status').lower() == 'complete':
        return('predictor')
    if state.get('validation_status') == None:
        return('Validator')
    
def intent_rout(state: AgentState):
    if state.get('user_intent').lower() == 'greeting':
        return (END)
    if state.get('user_intent').lower() == 'property':
        return ('extractor')
    if state.get('user_intent').lower() == 'research':
        return ('research')

workflow=StateGraph(AgentState)

workflow.add_node('intent',classify_intent)
workflow.add_node("extractor",extract_property_info)
workflow.add_node("predictor",Predictor)
workflow.add_node("responder",Generate_Response)
workflow.add_node('Rental',Rental_calculator)
workflow.add_node('Validator', validate_inputs)
workflow.add_node('research',research)

workflow.add_edge(START,'intent')
workflow.add_conditional_edges('intent',intent_rout)
workflow.add_edge("research", END)
workflow.add_edge("extractor","Validator")
workflow.add_conditional_edges("Validator",rout_validator)
workflow.add_edge("predictor","Rental")
workflow.add_edge("Rental","responder")
workflow.add_edge('responder',END)

real_estate_agent=workflow.compile(checkpointer=memory)

