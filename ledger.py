"""Conservative shared monthly CNY ledger. Unknown attempts keep their reserve."""
import asyncio
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import uuid

CST = timezone(timedelta(hours=8))
LIMIT = 50_000_000  # micro-yuan; never floats for budget decisions


class BudgetDenied(RuntimeError):
    pass


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS charges (id TEXT PRIMARY KEY, month TEXT NOT NULL, amount INTEGER NOT NULL, state TEXT NOT NULL)')

    @staticmethod
    def month(now=None):
        value = now or datetime.now(CST)
        if value.tzinfo is None:
            raise ValueError('Aware clock required')
        return value.astimezone(CST).strftime('%Y-%m')

    def _reserve(self, amount, now):
        if not isinstance(amount, int) or not 0 < amount <= LIMIT:
            raise BudgetDenied('本次请求超过可预留额度。')
        month = self.month(now)
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            used = db.execute('SELECT COALESCE(sum(amount),0) FROM charges WHERE month=?', (month,)).fetchone()[0]
            if used + amount > LIMIT:
                raise BudgetDenied('本月API预算已达50元保护线，请下月再用付费模型。')
            token = uuid.uuid4().hex
            db.execute("INSERT INTO charges VALUES(?,?,?,'reserved')", (token, month, amount))
            return token

    async def reserve(self, amount, now=None):
        return await asyncio.to_thread(self._reserve, amount, now)

    def _settle(self, token, amount):
        if not isinstance(amount, int) or amount < 0:
            raise BudgetDenied('API用量异常，保留本次预留额度。')
        with closing(sqlite3.connect(self.path, timeout=10)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT amount,state FROM charges WHERE id=?', (token,)).fetchone()
            if row is None or row[1] != 'reserved':
                raise BudgetDenied('预算记录不匹配。')
            if amount > row[0]:
                # An upstream contract violation freezes this month. Never hide
                # unexpectedly high usage by silently capping the recorded cost.
                month = db.execute('SELECT month FROM charges WHERE id=?', (token,)).fetchone()[0]
                db.execute("INSERT OR IGNORE INTO charges VALUES(?,?,?,'frozen')", ('freeze-'+month, month, LIMIT))
                db.execute("UPDATE charges SET amount=?,state='overrun' WHERE id=?", (amount, token))
                return False
            db.execute("UPDATE charges SET amount=?,state='settled' WHERE id=?", (amount, token))
            return True

    async def settle(self, token, amount):
        return await asyncio.to_thread(self._settle, token, amount)

    def seed(self, name, amount, month):
        assert isinstance(amount, int) and 0 <= amount <= LIMIT
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT OR IGNORE INTO charges VALUES(?,?,?,'prior_trial')", (name, month, amount))

    def total(self, now=None):
        with closing(sqlite3.connect(self.path)) as db:
            return db.execute('SELECT COALESCE(sum(amount),0) FROM charges WHERE month=?', (self.month(now),)).fetchone()[0]
