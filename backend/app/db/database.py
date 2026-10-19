from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

from app.core.config import get_settings

settings = get_settings()

database_url = URL.create(
    drivername="postgresql+psycopg",
    username=settings.postgres_user,
    password=settings.postgres_password.get_secret_value(),
    host=settings.postgres_host,
    port=settings.postgres_port,
    database=settings.postgres_db
)

engine = create_engine(
    database_url,
    pool_pre_ping=True,
    connect_args={"connect_timeout":5}
)

def check_database_connection() -> int:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))
        return result.scalar_one()