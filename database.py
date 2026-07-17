"""
database.py

SQLite + SQLAlchemy ORM setup for Cricket IQ. Four tables:

  matches      - one row per match seen (id, name, type, venue, date, status)
  players      - one row per player (id, name, team, role)
  innings      - one row per innings (match_id, team, runs, wickets, overs)
  predictions  - one row per prediction the API makes, so Match History
                 (dashboard.py Page 5) can compare predicted vs actual later

Everything is defined with SQLAlchemy's declarative ORM so the rest of the
app works with plain Python objects instead of hand-written SQL.
"""

import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DB_PATH = os.path.join(os.path.dirname(__file__), "cricket_iq.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# check_same_thread=False: FastAPI can handle a request on a different
# thread than the one that created the engine; SQLite's default is to
# forbid that, so this opts back in (safe here since each request gets
# its own short-lived Session, not a long-held shared connection).
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Match(Base):
    __tablename__ = "matches"

    id = Column(String, primary_key=True)  # CricAPI's match id (not auto-increment - it's already unique)
    name = Column(String)
    type = Column(String)  # t20 / odi / test
    venue = Column(String)
    date = Column(String)
    status = Column(String)

    innings = relationship("Innings", back_populates="match", cascade="all, delete-orphan")


class Player(Base):
    __tablename__ = "players"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    team = Column(String)
    role = Column(String)  # batter / bowler / all-rounder / wicketkeeper


class Innings(Base):
    __tablename__ = "innings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String, ForeignKey("matches.id"))
    team = Column(String)
    runs = Column(Integer, default=0)
    wickets = Column(Integer, default=0)
    overs = Column(Float, default=0.0)

    match = relationship("Match", back_populates="innings")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String, ForeignKey("matches.id"), nullable=True)
    predicted_score = Column(Integer, nullable=True)
    actual_score = Column(Integer, nullable=True)     # filled in later, once the match ends
    win_prediction = Column(Float, nullable=True)      # batting team's predicted win %
    win_actual = Column(Integer, nullable=True)        # 1/0, filled in later once the result is known
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Creates the SQLite file and all tables if they don't exist yet.
    Safe to call every time the API starts - existing tables are left alone."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency-style helper: yields a Session, always closes it
    afterwards even if the request raised. (api.py currently opens/closes
    sessions directly for simplicity, but this is here for routes that
    want the `Depends(get_db)` pattern.)"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # Run with: python database.py  -  creates cricket_iq.db and prints the tables.
    init_db()
    print(f"Database ready at {DB_PATH}")
    print("Tables:", list(Base.metadata.tables.keys()))
