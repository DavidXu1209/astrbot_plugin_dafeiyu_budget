"""Conservative shared monthly CNY ledger. Unknown attempts keep their reserve."""
import asyncio
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import random
import re
import sqlite3
import uuid

CST = timezone(timedelta(hours=8))
LIMIT = 50_000_000  # micro-yuan; never floats for budget decisions
SPEAKING_KEEP = 15_000_000  # stay able to talk after a treat
MILK_TEA = 15_000_000
COFFEE = 20_000_000
TREAT_PURSE = LIMIT - SPEAKING_KEEP
MAX_TREAT_CHANCE = 4000  # basis points; a full purse is a 40% chance, not a promise
DRINKS = {'milk_tea': MILK_TEA, 'coffee': COFFEE}
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
            db.execute(
                'CREATE TABLE IF NOT EXISTS treats ('
                'month TEXT NOT NULL, user_key TEXT NOT NULL, kind TEXT NOT NULL, '
                'amount INTEGER NOT NULL, state TEXT NOT NULL, '
                'PRIMARY KEY (month, user_key))'
            )

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
                raise BudgetDenied('大肥鱼钱包本月 50 元已经用完，请下月再用。')
            used = db.execute(
                'SELECT COALESCE(sum(amount),0) FROM charges WHERE month=?', (month,)
            ).fetchone()[0]
            if used + amount > LIMIT:
                raise BudgetDenied('大肥鱼钱包本月 50 元已经用完，请下月再用。')
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
            or name.startswith('treat-')
            or len(name) > 200
            or not self._amount(amount, allow_zero=True)
            or not isinstance(month, str)
            or _MONTH.fullmatch(month) is None
            or not 1 <= int(month[5:7]) <= 12
        ):
            raise ValueError('invalid prior-trial seed')
        with closing(self._connect()) as db, db:
            db.execute("INSERT OR IGNORE INTO charges VALUES(?,?,?,'prior_trial')", (name, month, amount))

    def _summary(self, now):
        month = self.month(now)
        with closing(self._connect()) as db:
            rows = db.execute(
                'SELECT state, COALESCE(sum(amount),0) FROM charges WHERE month=? GROUP BY state',
                (month,),
            ).fetchall()
            winner = db.execute(
                "SELECT user_key, kind, amount FROM treats WHERE month=? AND state='granted'",
                (month,),
            ).fetchone()
        states = {state: amount for state, amount in rows}
        total = sum(states.values())
        frozen = 'frozen' in states
        remaining = 0 if frozen or total >= LIMIT else LIMIT - total
        treat_for = None
        if winner is not None:
            treat_for = {'user': winner[0], 'kind': winner[1], 'amount': winner[2]}
        return {
            'month': month, 'limit': LIMIT, 'states': states, 'total': total,
            'frozen': frozen, 'remaining': remaining, 'treat_for': treat_for,
        }

    def _ask_treat(self, user_key, kind, now, roll):
        """One ask per person per month. At most one granted drink, and only by chance."""
        if kind not in DRINKS:
            raise ValueError('unknown drink')
        amount = DRINKS[kind]
        if isinstance(roll, bool) or not isinstance(roll, int) or not 0 <= roll < 10000:
            raise ValueError('treat roll must be an integer from 0 through 9999')
        if (
            not isinstance(user_key, str)
            or not user_key.strip()
            or len(user_key) > 64
            or any(char in user_key for char in '\n\r')
        ):
            return {'result': 'bad_user', 'chance': 0, 'amount': amount, 'kind': kind}
        user_key = user_key.strip()
        month = self.month(now)
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute(
                "SELECT 1 FROM charges WHERE month=? AND state='frozen' LIMIT 1", (month,)
            ).fetchone():
                return {'result': 'frozen', 'chance': 0, 'amount': amount, 'kind': kind}
            prior = db.execute(
                'SELECT state FROM treats WHERE month=? AND user_key=?', (month, user_key)
            ).fetchone()
            if prior is not None:
                return {'result': 'asked', 'chance': 0, 'amount': amount, 'kind': kind}
            if db.execute('SELECT 1 FROM charges WHERE id=?', (f'treat-{month}',)).fetchone():
                return {'result': 'taken', 'chance': 0, 'amount': amount, 'kind': kind}
            used = db.execute(
                'SELECT COALESCE(sum(amount),0) FROM charges WHERE month=?', (month,)
            ).fetchone()[0]
            available = LIMIT - used - SPEAKING_KEEP
            if available < amount:
                return {'result': 'broke', 'chance': 0, 'amount': amount, 'kind': kind}
            chance = MAX_TREAT_CHANCE * available // TREAT_PURSE
            if roll >= chance:
                db.execute(
                    "INSERT INTO treats VALUES(?,?,?,?,'missed')",
                    (month, user_key, kind, amount),
                )
                return {'result': 'missed', 'chance': chance, 'amount': amount, 'kind': kind}
            db.execute(
                "INSERT INTO charges VALUES(?,?,?,'treat')",
                (f'treat-{month}', month, amount),
            )
            db.execute(
                "INSERT INTO treats VALUES(?,?,?,?,'granted')",
                (month, user_key, kind, amount),
            )
            return {'result': 'granted', 'chance': chance, 'amount': amount, 'kind': kind}

    async def ask_treat(self, user_key, kind, now=None, roll=None):
        if roll is None:
            roll = random.randrange(10000)
        return await asyncio.to_thread(self._ask_treat, user_key, kind, now, roll)

    async def summary(self, now=None):
        return await asyncio.to_thread(self._summary, now)

    def total(self, now=None):
        with closing(self._connect()) as db:
            return db.execute(
                'SELECT COALESCE(sum(amount),0) FROM charges WHERE month=?', (self.month(now),)
            ).fetchone()[0]
