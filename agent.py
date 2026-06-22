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
import json

memory=MemorySaver()
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_ENDPOINT"]=os.getenv("LANGCHAIN_ENDPOINT")
os.environ["LANGCHAIN_TRACING_V2"]=os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

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
    Baths: Optional[int] = Field(
        default=2, description="the number of bathrooms. if not specified assume bedrooms+1"
    )
    Type: Optional[str] =Field(
        default=None,
        description="the type of property (eg: Villa,Apartment). default set to None if not specified"
    )
    Area_in_sqft: Optional[float] = Field(
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
    City: Optional[str] = Field(
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
    Location=state.get('Location')
    Beds=state.get('Beds')
    Type=state.get('Type')
    classification_prompt = f"""Analyze the user query: '{query}'.
    

Existing property details already collected: Location={Location}, Beds={Beds}, Type={Type}

- If existing details are already filled, it means a property conversation is in progress — classify as 'property'
- If they are looking to rent/find a specific property, classify as 'property'
- If they are asking for market news, trends or general real estate information, classify as 'research'
- If it's just a greeting, classify as 'greeting'
- If the user is asking for contact details or how to proceed with renting, classify as 'contact'

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
    elif 'contact' in intent.content.lower():
        return {'user_intent':'contact'}
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

    context = f"""
Extract property details from the user message and return a JSON object.

Existing property details: Beds={Beds}, Baths={Baths}, Type={Type}, Area_in_sqft={Area_in_sqft}, 
Furnishing={Furnishing}, Location={Location}, City={City}

New user message: {text}
If the new user message explicitly mentions any of the property details (Beds, Baths, Type, Area_in_sqft, Furnishing, Location, City), update that field with the new value.
If the user is saying fully furnished or partially furnished, set Furnishing to Furnished. If they say unfurnished, set it to Unfurnished.
Only update a field if the new message explicitly mentions it. If a field already has a value keep it exactly as is. Never set a field back to None.

Return ONLY a valid JSON object with these exact keys: Beds, Baths, Type, Area_in_sqft, Furnishing, Location, City.
"""
    structured_llm=llm.with_structured_output(PropertyExtraction,method='json_mode')
    extracted_content=structured_llm.invoke(context)
    
    beds = extracted_content.Beds or state.get('Beds')
    baths = extracted_content.Baths or state.get('Baths') or (beds + 1 if beds else 2)
    area = extracted_content.Area_in_sqft or state.get('Area_in_sqft') or (beds * 750 + 24 if beds else 750)
    return {
    'Beds': beds,
    'Baths': baths,
    'Type': (extracted_content.Type or state.get('Type') or '').title() or None,
    'Area_in_sqft': area,
    'Furnishing': extracted_content.Furnishing or state.get('Furnishing'),
    'Location': extracted_content.Location or state.get('Location'),
    'City': extracted_content.City or state.get('City') or 'Dubai'  # always falls back to Dubai
}

    
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
        prompt=f""" You are a professional UAE real estate advisor. Ask the client the following in a warm, natural, conversational way: {missing_str}. Keep it to one sentence.
        if the user has already provided some of the missing information in their previous query, do not ask for it again. Only ask for what is still missing. For example, if they already mentioned they want a furnished property, do not ask about furnishing again. Just ask about the remaining missing details."""
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
    total_amount=security_deposit+agent_commission
    
    return {"monthly_rent":Monthly_rent,'annual_rent':Annual_rent, 'security_deposit':security_deposit, 'agent_commission': agent_commission,'total_amount':total_amount}


def format_aed(amount):
    return f"AED {round(amount):,}"

def contact_details(state: AgentState):
    return {
        "final_response": (
            "You can contact Essa Insham here:\n"
            "- Mobile: +971 52 718 1331\n"
            "- LinkedIn: https://www.linkedin.com/in/essa-insham\n\n"
            "If you'd like, I can also help you prepare the key details to share before contacting the agent."
        )
    }

def Generate_Response(state: AgentState):
    query=state['user_query']
    location=state['Location']
    price=state['predicted_price']
    monthly_rent=state['monthly_rent']
    annual_rent=state['annual_rent']
    security_deposit=state['security_deposit']
    agent_commission=state['agent_commission']
    total_amount=state.get('total_amount')

    annual_rent_display = format_aed(annual_rent)
    monthly_rent_display = format_aed(monthly_rent)
    security_deposit_display = format_aed(security_deposit)
    agent_commission_display = format_aed(agent_commission)
    total_amount_display = format_aed(total_amount)

    prompt=f""" You are a warm, helpful, and professional UAE real estate rental advisor interacting with a client in a conversational chat interface.

Tone and format rules:
- Aim for 180-250 words. Never exceed 300 words.
- Start naturally by acknowledging the user's request in one short sentence. Do not introduce the agent by name unless the user asks for contact details or a viewing.- Do not format the response as an email.
- Include one short neighborhood insight when a location is known like a short paragraph. The insight should explain why the area may fit the user’s lifestyle or budget, without exaggerating or making unsupported claims.
- Do not use generic praise like "beautiful", "amazing", "perfect", or "luxury" unless the specific location context supports it. Prefer practical details: commute access, family-friendliness, waterfront lifestyle, schools, malls, quieter community, business districts, or value-for-money.
- add the paragraph about the neighbourhood in a single paragraph, not a list before the financial breakdown as a separate paragraph.
- Do not use greetings like "Dear client" or sign-offs like "Best regards".
- Do not repeat financial figures, contact details, assumptions, or recommendations.
- Avoid generic repeated phrases like "popular area". Give one practical reason the location fits the user.

Pricing policy:
- Describe the model output as an estimated annual rent generated from the trained property-price model using the user's requirements.
- Do not claim this guarantees live market availability.
- If the user mentioned a budget or expected rent, compare it briefly. Otherwise do not invent a comparison.
- if providing a financial breakdown, use the currency AED and round all figures to the nearest whole number.
- Always format monetary amounts in AED using rounded whole numbers and digit grouping, e.g., AED 223,513.
- This is an estimate from the trained property-price model, so treat it as guidance rather than live availability.

User context:
- User query: "{query}"
- Location: {location}
- Estimated annual rent from model: {price}

Financial breakdown:
Display exactly once using these bullets:
* Annual Rent: {annual_rent_display}
* Monthly Rent: {monthly_rent_display}
* Security Deposit: {security_deposit_display}
* Agent Commission: {agent_commission_display}
* Estimated Upfront Cost: {total_amount_display}

- If the user is exploring prices or areas, ask one useful refinement question.
- If the user shows action intent, such as asking about viewing, availability, negotiation, agent help, or next steps, ask whether they would like the agent contact details.
    """

    response=llm.invoke(prompt)

    return {'final_response':response.content}


def research(state: AgentState):
    query=state['user_query']
    search_response=search_tool.invoke({"query":query})
    results=search_response.get('results', []) if isinstance(search_response, dict) else search_response
    if not results:
        return {'final_response':"No results found for your query. Try rephrasing your question or ask about a different topic."}
    
    blocked_domains = ["instagram.com", "reddit.com", "tiktok.com", "facebook.com", "x.com", "twitter.com", "linkedin.com"]
    source_lines = []
    informal_lines = []
    for r in results:
        title = r.get('title', 'Untitled')
        url = r.get('url', 'No URL')
        if not url:
            continue
        line = f"- {title}: {url}"
        if any(domain in url for domain in blocked_domains):
            informal_lines.append(line)
        else:
            source_lines.append(line)

    source_lines=source_lines[:3]  # Limit to top 3 sources
    informal_lines=informal_lines[:2]  # Limit to top 2 informal sources


    combined_context = "\n\n".join(
        f"""source:{i+1}\n
        Title: {r.get('title', 'Untitled')}\n
        URL: {r.get('url', 'No URL')}\n
        content: {r.get('content', 'No content available')}\n"""
        for i, r in enumerate(results)
    )
    today_str = date.today().strftime("%B %d, %Y")
    prompt = f"""
You are a UAE real estate market advisor. Today's date is {today_str}.

A client asked: "{query}"

Here is recent information gathered from multiple search results:
{combined_context}

Write a clear, balanced market summary that directly answers the client's question.

Rules:
- Use only the information provided in the search results above.
- Keep the answer professional and concise, around 3-5 sentences.
- Start with "As of {today_str}, ..."
- Synthesize the sources into a coherent answer instead of quoting one source.
- If stronger sources disagree, mention both perspectives briefly.
- Treat Instagram, Reddit, TikTok, Facebook, and X as informal/unverified sources.
- Do not use informal/unverified sources as the main basis for factual claims.
- If informal sources suggest a different view, describe it cautiously as informal or unverified.
- Do not include source URLs in the answer body.
- Do not include a "Sources" section.
- Do not list source titles in the answer body.
- The application will add the source list separately after your answer.
"""
    


    response=llm.invoke(prompt)
    final=response.content
    if source_lines:
        final += "\n\nSources:\n" + "\n".join(source_lines)
    if informal_lines:
        final += "\n\nInformal/unverified sources:\n" + "\n".join(informal_lines)
    

    return {'final_response':final}

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
    if state.get('user_intent').lower() == 'contact':
        return ('contact')
    else:
        return (END)

workflow=StateGraph(AgentState)

workflow.add_node('intent',classify_intent)
workflow.add_node("extractor",extract_property_info)
workflow.add_node("predictor",Predictor)
workflow.add_node("responder",Generate_Response)
workflow.add_node('Rental',Rental_calculator)
workflow.add_node('Validator', validate_inputs)
workflow.add_node('contact',contact_details)
workflow.add_node('research',research)

workflow.add_edge(START,'intent')
workflow.add_conditional_edges('intent',intent_rout)
workflow.add_edge("research", END)
workflow.add_edge("extractor","Validator")
workflow.add_conditional_edges("Validator",rout_validator)
workflow.add_edge("predictor","Rental")
workflow.add_edge("Rental","responder")
workflow.add_edge('contact',END)
workflow.add_edge('responder',END)

real_estate_agent=workflow.compile(checkpointer=memory)

