from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./cartguard.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)

Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    time_spent = Column(Integer)
    cart_value = Column(Integer)
    num_items = Column(Integer)
    logged_in = Column(Integer)
    discount_applied = Column(Integer)
    previous_purchases = Column(Integer)
    device_mobile = Column(Integer)
    probability = Column(Float)
    risk_level = Column(String)


Base.metadata.create_all(bind=engine)
