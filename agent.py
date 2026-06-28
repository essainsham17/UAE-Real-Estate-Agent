from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import AgentState
from config import memory
from nodes import (
    classify_intent, extract_property_info, validate_inputs,
    predictor, rental_calculator, generate_response,
    research, legal_advisor, contact_details,
    intent_rout, rout_validator
)

workflow=StateGraph(AgentState)

workflow.add_node('intent',classify_intent)
workflow.add_node("extractor",extract_property_info)
workflow.add_node("predictor",predictor)
workflow.add_node("responder",generate_response)
workflow.add_node('rental',rental_calculator)
workflow.add_node('validator', validate_inputs)
workflow.add_node('contact',contact_details)
workflow.add_node('research',research)
workflow.add_node('legal',legal_advisor)

workflow.add_edge(START,'intent')
workflow.add_conditional_edges('intent',intent_rout)
workflow.add_edge("research", END)
workflow.add_edge("extractor","validator")
workflow.add_conditional_edges("validator",rout_validator)
workflow.add_edge("predictor","rental")
workflow.add_edge("rental","responder")
workflow.add_edge('contact',END)
workflow.add_edge('responder',END)
workflow.add_edge('legal',END)

real_estate_agent=workflow.compile(checkpointer=memory)

