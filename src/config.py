import os
from dotenv import load_dotenv

load_dotenv()

MYSQL_ROOT_PASSWORD = os.getenv("MYSQL_ROOT_PASSWORD", "your_password")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "logs_db")

# Add other configurations as needed


