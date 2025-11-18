from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.db.session import SessionLocal
from app.db.models import Rating

def main():
    session = SessionLocal()
    deleted = session.query(Rating).delete()  #DELETE FROM ratings;
    session.commit()
    session.close()
    print(f"Deleted {deleted} rating rows.")

if __name__ == "__main__":
    main()
