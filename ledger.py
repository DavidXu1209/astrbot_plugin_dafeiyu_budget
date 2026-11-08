"""Conservative shared monthly CNY ledger. Unknown attempts keep their reserve."""
import asyncio
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import sqlite3
import uuid

CST = timezone(timedelta(hours=8))
LIMIT = 50_000_000  # micro-yuan; never floats for budget decisions
_MONTH = re.compile(r'\d{4}-\d{2}$')


class BudgetDenied(RuntimeError):
    pass


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute(
                'CREATE TABLE IF NOT EXISTS charges ('
                'id TEXT PRIMARY KEY, month TEXT NOT NULL, amount INTEGER NOT NULL, state TEXT NOT NULL)'
            )
            db.execute('CREATE INDEX IF NOT EXISTS charges_month ON charges(month)')

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute('PRAGMA busy_timeout=10000')
        return db

    @staticmethod
    def month(now=None):
        value = now or datetime.now(CST)
        if value.tzinfo is None:
            raise ValueError('Aware clock required')
        return value.astimezone(CST).strftime('%Y-%m')

    @staticmethod
    def _amount(amount, *, allow_zero):
        if isinstance(amount, bool) or not isinstance(amount, int):
            return False
        lower = 0 if allow_zero else 1
        return lower <= amount <= LIMIT

    def _reserve(self, amount, now):
        if not self._amount(amount, allow_zero=False):
            raise BudgetDenied('本次请求超过可预留额度。')
        month = self.month(now)
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute(
                "SELECT 1 FROM charges WHERE month=? AND state='frozen' LIMIT 1", (month,)
            ).fetchone():
                raise BudgetDenied('本月API预算已达50元保护线，请下月再用付费模型。')
            used = db.execute(
                'SELECT COALESCE(sum(amount),0) FROM charges WHERE month=?', (month,)
            ).fetchone()[0]
            if used + amount > LIMIT:
                raise BudgetDenied('本月API预算已达50元保护线，请下月再用付费模型。')
            token = uuid.uuid4().hex
            db.execute("INSERT INTO charges VALUES(?,?,?,'reserved')", (token, month, amount))
            return token

    async def reserve(self, amount, now=None):
        return await asyncio.to_thread(self._reserve, amount, now)

    def _mark_frozen(self, db, month):
        # Full-cap marker, not a second copy of the real spend. reserve() also
        # rejects the month when this row exists, so a later reader that only
        # sums amounts still cannot reopen the month.
        db.execute(
            "INSERT OR IGNORE INTO charges VALUES(?,?,?,'frozen')",
            ('freeze-'+month, month, LIMIT),
        )

    def _settle(self, token, amount):
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise BudgetDenied('API用量异常，保留本次预留额度。')
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT amount,state,month FROM charges WHERE id=?', (token,)).fetchone()
            if row is None or row[1] != 'reserved':
                raise BudgetDenied('预算记录不匹配。')
            if amount > row[0]:
                # An upstream contract violation freezes this month. Never hide
                # unexpectedly high usage by silently capping the recorded cost.
                self._mark_frozen(db, row[2])
                db.execute("UPDATE charges SET amount=?,state='overrun' WHERE id=?", (amount, token))
                return False
            db.execute("UPDATE charges SET amount=?,state='settled' WHERE id=?", (amount, token))
            return True

    async def settle(self, token, amount):
        return await asyncio.to_thread(self._settle, token, amount)

    def _freeze(self, token):
        """Block the rest of the month when usage cannot be verified."""
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state,month FROM charges WHERE id=?', (token,)).fetchone()
            if row is None or row[0] != 'reserved':
                raise BudgetDenied('预算记录不匹配。')
            self._mark_frozen(db, row[1])
            return False

    async def freeze(self, token):
        return await asyncio.to_thread(self._freeze, token)

    def seed(self, name, amount, month):
        if (
            not isinstance(name, str)
            or not name
            or name.startswith('freeze-')
            or len(name) > 200
            or not self._amount(amount, allow_zero=True)
            or not isinstance(month, str)
            or _MONTH.fullmatch(month) is None
            or not 1 <= int(month[5:7]) <= 12
        ):
            raise ValueError('invalid prior-trial seed')
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR IGNORE INTO charges VALUES(?,?,?,'prior_trial')", (name, month, amount))

    def total(self, now=None):
        with closing(self._connect()) as db:
            return db.execute(
                'SELECT COALESCE(sum(amount),0) FROM charges WHERE month=?', (self.month(now),)
            ).fetchone()[0]
