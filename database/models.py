# database/models.py
from sqlalchemy import Column, Integer, String, Date, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.orm import relationship
from datetime import date
from pgvector.sqlalchemy import Vector
from .database import Base


class Person(Base):
    __tablename__ = "persons"

    id = Column(Integer, primary_key=True)

    embeddings = relationship("PersonEmbedding", back_populates="person", cascade="all, delete")
    attendance_records = relationship("Attendance", back_populates="person", cascade="all, delete")



class PersonEmbedding(Base):
    __tablename__ = "person_embeddings"

    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), primary_key=True)  # بقا primary key
    embedding = Column(Vector(256), nullable=False)

    # اتمسح الـ position والـ PrimaryKeyConstraint

    person = relationship("Person", back_populates="embeddings")




class Attendance(Base):
    __tablename__ = "attendance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(Integer, ForeignKey("persons.id", ondelete="CASCADE"), nullable=False)
    attendance_date = Column(Date, default=date.today)

    person = relationship("Person", back_populates="attendance_records", foreign_keys=[person_id])
