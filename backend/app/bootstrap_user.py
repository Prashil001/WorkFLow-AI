"""Provision the one configured API identity for this single-user demo."""

from sqlalchemy import select

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.models.user import User


def provision_configured_user() -> str:
    settings = get_settings()
    configured_id = settings.workflow_user_id
    configured_email = settings.workflow_user_email

    with SessionLocal.begin() as session:
        user = session.get(User, configured_id)
        if user is not None:
            if user.email != configured_email:
                raise ValueError(
                    "Configured user ID is already assigned to another email"
                )
            return str(configured_id)

        email_owner = session.scalar(
            select(User).where(User.email == configured_email)
        )
        if email_owner is not None:
            raise ValueError(
                "Configured email is already assigned to another user ID"
            )

        session.add(User(
            id=configured_id,
            email=configured_email,
        ))

    return str(configured_id)


def main() -> None:
    user_id = provision_configured_user()
    print(f"Configured API identity is ready: {user_id}")


if __name__ == "__main__":
    main()
