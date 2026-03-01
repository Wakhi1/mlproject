from sqlalchemy import Column, Integer, String, Boolean
from app.database import Base

# SQLAlchemy User model
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    full_name = Column(String(100))
    hashed_password = Column(String(255), nullable=False)
    disabled = Column(Boolean, default=False)

# Pydantic schemas for request/response
from pydantic import BaseModel, EmailStr

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None

class UserCreate(UserBase):
    password: str

class UserInDB(UserBase):
    id: int
    disabled: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class ComplianceInput(BaseModel):
    taxpayer_type: str
    region: str
    industry_sector: str
    years_registered: int
    annual_turnover_szl: float
    vat_registered: bool
    paye_registered: bool
    num_employees_declared: int
    filings_due_last_12m: int
    filings_submitted_last_12m: int
    late_filings_count: int
    amended_returns_count: int
    outstanding_tax_szl: float
    penalty_count: int
    prior_audit_flag: bool
    prior_audit_finding: bool
    days_since_last_payment: int
    payment_plan_active: bool
    cross_border_transactions: bool

    class Config:
        json_schema_extra = {
            "example": {
                "taxpayer_type": "Company",
                "region": "Manzini",
                "industry_sector": "Retail",
                "years_registered": 5,
                "annual_turnover_szl": 1200000,
                "vat_registered": True,
                "paye_registered": True,
                "num_employees_declared": 30,
                "filings_due_last_12m": 12,
                "filings_submitted_last_12m": 10,
                "late_filings_count": 3,
                "amended_returns_count": 2,
                "outstanding_tax_szl": 78000,
                "penalty_count": 2,
                "prior_audit_flag": True,
                "prior_audit_finding": True,
                "days_since_last_payment": 95,
                "payment_plan_active": False,
                "cross_border_transactions": False
            }
        }

class ComplianceOutput(BaseModel):
    prediction: int
    risk_probability: float
    risk_level: str