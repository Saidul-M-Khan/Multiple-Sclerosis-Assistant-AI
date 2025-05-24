from app.database import engine
from app.models import Base
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    """Initialize the database by dropping and recreating all tables."""
    try:
        # Drop all tables
        Base.metadata.drop_all(bind=engine)
        logger.info("Dropped all existing tables")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Created all tables successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    init_db()
    print("Database initialized successfully!") 