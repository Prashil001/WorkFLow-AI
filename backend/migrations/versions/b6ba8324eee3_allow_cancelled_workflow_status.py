"""allow cancelled workflow status

Revision ID: b6ba8324eee3
Revises: 6118f2d9c9d5
Create Date: 2026-10-19 19:16:53.084579

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = 'b6ba8324eee3'
down_revision: Union[str, Sequence[str], None] = '6118f2d9c9d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_workflows_status",
        "workflows",
        type_="check",
    )

    op.create_check_constraint(
        "ck_workflows_status",
        "workflows",
        "status IN ("
        "'PLANNING', 'RUNNING', "
        "'WAITING_FOR_INPUT', "
        "'WAITING_FOR_APPROVAL', "
        "'COMPLETED', 'PARTIAL_SUCCESS', "
        "'FAILED', 'CANCELLED'"
        ")",
    )


def downgrade() -> None:
    connection = op.get_bind()

    cancelled_count = connection.execute(
        text(
            "SELECT COUNT(*) FROM workflows "
            "WHERE status = 'CANCELLED'"
        )
    ).scalar_one()

    if cancelled_count > 0:
        raise RuntimeError(
            "Cannot downgrade while cancelled "
            "workflow records exist"
        )

    op.drop_constraint(
        "ck_workflows_status",
        "workflows",
        type_="check",
    )

    op.create_check_constraint(
        "ck_workflows_status",
        "workflows",
        "status IN ("
        "'PLANNING', 'RUNNING', "
        "'WAITING_FOR_INPUT', "
        "'WAITING_FOR_APPROVAL', "
        "'COMPLETED', 'PARTIAL_SUCCESS', "
        "'FAILED'"
        ")",
    )
