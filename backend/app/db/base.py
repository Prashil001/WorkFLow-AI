from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass


class RagBase(DeclarativeBase):
    """PostgreSQL-only tables (pgvector, tsvector).

    Kept off ``Base.metadata`` so portable schemas such as the SQLite test
    fixtures can still ``create_all`` the core workflow tables.
    """