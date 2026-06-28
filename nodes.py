import pandas as pd
import numpy as np
from datetime import date
from langgraph.graph import END
from state import AgentState, PropertyExtraction
from config import llm, retriever, search_tool, prediction_model




def classify_intent(state: AgentState):
    query=state.get('user_query')
    Location=state.get('Location')
    Beds=state.get('Beds')
    Type=state.get('Type')
    classification_prompt = f"""
You are an expert intent classification assistant for a real estate platform.
Analyze the user's latest query alongside the state of the collected property details.

Context:
User Query: "{query}"
Collected Details: Location="{Location}", Beds="{Beds}", Type="{Type}"

Rules:
1. Evaluate the categories in the exact order listed below. Assign the first matching label.
2. Your response MUST be EXACTLY ONE WORD from the allowed categories list.
3. Do not include any punctuation, spaces, introductions, or reasoning text. 
4. Failure to output exactly one word from the list will break the system.

Categories:
- property: The user is looking to rent/find a specific property, OR ANY of the Collected Details (Location, Beds, Type) are already filled/not empty (indicating a property conversation is currently in progress).
- research: The user is asking for market news, trends, statistics, prices, or general real estate information.
- contact: The user is asking for contact details, phone numbers, emails, or how to proceed with booking/renting.
- legal: The user is asking for legal advice, contract rules, tenancy laws, disputes, or regulations.
- greeting: The user's query is just a greeting (e.g., "hi", "hello") or general pleasantry with no active property context.

Examples:
User Query: "Okay, sounds good."
Collected Details: Location="Dubai Marina", Beds="2", Type="Apartment"
Output: property

User Query: "I am looking for a 3-bedroom villa to rent."
Collected Details: Location="None", Beds="None", Type="None"
Output: property

User Query: "What is the average ROI for studio apartments downtown?"
Collected Details: Location="None", Beds="None", Type="None"
Output: research

User Query: "Can I get the phone number of the agent handling this listing?"
Collected Details: Location="None", Beds="None", Type="None"
Output: contact

User Query: "Is it legal for my landlord to increase rent without a 90-day notice?"
Collected Details: Location="None", Beds="None", Type="None"
Output: legal

User Query: "Hello there, good morning!"
Collected Details: Location="None", Beds="None", Type="None"
Output: greeting

Output format:
[Only the single-word category label here, nothing else]
"""

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
    elif 'legal' in intent.content.lower():
        return {'user_intent':'legal'}
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


def predictor(state: AgentState):
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


def rental_calculator(state: AgentState):
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

def generate_response(state: AgentState):
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
    uae_keywords = ["uae", "dubai", "abu dhabi", "sharjah", "ajman", "ras al khaimah", "fujairah", "umm al quwain", "united arab emirates",'rak']
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in uae_keywords):
        pass  # Query is relevant to UAE real estate
    else:
        query = f"{query} in UAE real estate market"


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

def legal_advisor(state: AgentState):
    query=state['user_query']
    retrieve_docs=retriever.invoke(query)
    if not retrieve_docs:
        return {'final_response':"I could not find any relevant legal information for your query. Please consult a qualified legal professional for specific advice."}
    legal_context=' '.join([doc.page_content for doc in retrieve_docs])
    prompt=f"""
You are a UAE real estate legal advisor. A client asked: "{query}"
Here is relevant information from recent documents:
{legal_context}
Write a clear, concise answer to the client's question based on the information provided. If the information is insufficient, respond with a warm, professional message indicating that you cannot provide a definitive answer and suggest
    that the client consults a qualified legal professional for specific advice. cite the document title and URL in your answer if you reference any information from the documents. Do not make up information or provide legal advice beyond what is in the documents."""
    response=llm.invoke(prompt)
    return {'final_response':response.content}

def rout_validator(state: AgentState):
    if state.get('validation_status') == None:
        return('Validator')
    if state.get('validation_status').lower() == 'incomplete':
        return(END)
    if state.get('validation_status').lower() == 'complete':
        return('predictor')
    
def intent_rout(state: AgentState):
    if state.get('user_intent').lower() == 'greeting':
        return (END)
    if state.get('user_intent').lower() == 'property':
        return ('extractor')
    if state.get('user_intent').lower() == 'research':
        return ('research')
    if state.get('user_intent').lower() == 'contact':
        return ('contact')
    if state.get('user_intent').lower() == 'legal':
        return ('legal')
    else:
        return (END)