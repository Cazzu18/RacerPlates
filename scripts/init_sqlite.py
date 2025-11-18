import sys
from pathlib import Path

#ensuriing backend/app is on the Python path so `app.*` imports work
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.models import Base
from app.db.session import engine

def main():
    Base.metadata.create_all(bind=engine)
    print("SQLite initialized!")
    
if __name__ == "__main__":
    main()
