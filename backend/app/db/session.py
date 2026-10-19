from collections.abc import Iterator

from sqlalchemy.orm import Session, sessionmaker

from app.db.database import engine

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)

def get_db_session() -> Iterator[Session]:
    with SessionLocal() as session:
        yield session