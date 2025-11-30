# # init_db.py
# from database.database import create_tables

# if __name__ == "__main__":
#     create_tables()
#     print(" Tables created successfully in PostgreSQL!")

# test_models.py - في نفس مستوى مجلد database
from database.models import Person, PersonEmbedding, Attendance
from database.database import create_tables

# الآن شغل الكود
create_tables()
print(" Tables created successfully in PostgreSQL")