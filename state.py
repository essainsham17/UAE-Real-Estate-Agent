from typing import TypedDict, Optional
from pydantic import BaseModel, Field

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
