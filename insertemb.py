# insertemb.py
import json
from sqlalchemy.orm import Session
from database.database import SessionLocal
from database.models import Person, PersonEmbedding

def insert_embeddings_from_json(json_file_path: str):
    """
    Insert person embeddings from JSON file into the database
    """
    # Read the JSON file
    with open(json_file_path, 'r') as f:
        embeddings_data = json.load(f)
    
    # Create database session
    db: Session = SessionLocal()
    
    try:
        # Process each person in the JSON
        for person_id, person_data in embeddings_data.items():
            print(f"Processing person: {person_id}")
            
            # Convert person_id to integer (e.g. "0001" -> 1)
            pid = int(person_id)

            # Check if person exists
            existing_person = db.query(Person).filter(Person.id == pid).first()

            if not existing_person:
                # Insert person using provided ID
                person = Person(id=pid)
                db.add(person)
                print(f"  - Inserted person with ID: {pid}")
            else:
                person = existing_person
                print(f"  - Person {pid} already exists")

            # Get embedding
            embedding_vector = person_data['embedding']

            # Create / update embedding
            existing_embedding = db.query(PersonEmbedding).filter(
                PersonEmbedding.person_id == pid
            ).first()

            if not existing_embedding:
                person_embedding = PersonEmbedding(
                    person_id=pid,
                    embedding=embedding_vector
                )
                db.add(person_embedding)
                print(f"  - Added embedding for person {pid}")
            else:
                existing_embedding.embedding = embedding_vector
                print(f"  - Updated embedding for person {pid}")

        # Commit all changes
        db.commit()
        print(f"\n✅ Successfully inserted {len(embeddings_data)} persons with embeddings!")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error occurred: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    insert_embeddings_from_json("P1_embeddings_arc_retina.json")
