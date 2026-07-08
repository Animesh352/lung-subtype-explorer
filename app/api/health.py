from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import check_db_connection, get_db

router = APIRouter()


@router.get("/health")
def health_check(db: Session = Depends(get_db)):
    db_ok = check_db_connection(db)
    return {
        "status": "ok",
        "database": "ok" if db_ok else "error",
    }
