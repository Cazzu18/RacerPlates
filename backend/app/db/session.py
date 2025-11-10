#Object Relational Mapper(orm) is a component that allows developers to interact with relational database using python objects, rather than raw SQL queries
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

#Sqlite neds check_same_thread=False for single-threaded FastApi dev servers
engine = create_engine(settings.DB_URL, connect_args={"check_same_thread":False} if settings.DB_URL.startswith("sqlite") else {})

#a configurable Session factory. Generates a new Session objects when called, creating them given config args established
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
