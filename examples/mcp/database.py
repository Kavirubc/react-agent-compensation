"""MongoDB database connection for Family Coordination MCP Server.

Requires MongoDB running on localhost:27017 or configure via environment:
- MONGODB_URL: MongoDB connection string
- DATABASE_NAME: Database name (default: family_coordination)
"""

import os

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "family_coordination")

client = MongoClient(MONGODB_URL)
db = client[DATABASE_NAME]

# Collections
family_members = db["family_members"]
tasks = db["tasks"]
pickups = db["pickups"]
cooking_tasks = db["cooking_tasks"]


def get_db():
    """Get database instance."""
    return db


def reset_all():
    """Reset all collections (for testing)."""
    family_members.delete_many({})
    tasks.delete_many({})
    pickups.delete_many({})
    cooking_tasks.delete_many({})
