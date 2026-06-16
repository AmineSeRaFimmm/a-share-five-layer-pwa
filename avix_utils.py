from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import akshare as ak
import numpy as np
import pandas as pd


DATA_DIR = Path("./data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
AVIX_INDEX_FILE = DATA_DIR / "avix_300_close_mid.csv"
AVIX_CHAIN_FILE = DATA_DIR / "avix_chain_snapshots.csv"
AVIX_HIST_FILE = DATA_DIR / "avix_300_hist_clean.csv"
AVIX_RAW_CLOSE_FILE = DATA_DIR / "avix_300_raw_close.csv"
AVIX_QVIX_FALLBACK_FILE = DATA_DIR / "avix_300_qvix_fallback.csv"
AVIX_CONTRACT_CACHE_FILE = DATA_DIR / "io_option_daily_cache.csv"


@dataclass
class TermVariance:
    expiry: pd.Timestamp
    dte: int
    t_year: float
    variance: float
    forward: float
    k0: float
    n_options: int
    n_puts: int
    n_calls: int
    rate: float
    quality: str
    note: str


class AVIX300CloseMid:
    """沪深300股指期权日频隐含波动率指数，AKShare mid-quote 研究版。"""

    def __init__(
        self,
        target_days: int = 30,
        min_dte: int = 7,
        default_rate: float = 0.02,
        min_term_options: int = 8,
    ):
        self.target_days = target_days
        self.min_dte = min_dte
        self.default_rate = default_rate
        self.min_term_options = min_term_options
        self._rate_points: list[tuple[int, float]] | None = None

    @staticmethod
    def _to_float(value: object) -> float:
        try:
            if value is None:
                return np.nan
            return float(value)
        except Exception:
            return np.nan

    @staticmethod
    def _to_int(value: object) -> int:
        try:
            if value is None or pd.isna(value):
                return 0
            return int(float(value))
        except Exception:
            return 0

    @staticmethod
    def _local_timestamp(value: datetime | None = None) -> pd.Timestamp:
        if value is None:
            value = datetime.now(ZoneInfo("Asia/Shanghai"))
        ts = pd.Timestamp(value)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts

    @staticmethod
    def _third_friday(year: int, month: int) -> pd.Timestamp:
        day = pd.Timestamp(year=year, month=month, day=1)
        fridays = []
        while day.month == month:
            if day.weekday() == 4:
                fridays.append(day)
            day += pd.Timedelta(days=1)
        return fridays[2]

    @staticmethod
    def _parse_io_month(month_symbol: str) -> tuple[int, int]:
        match = re.search(r"io(\d{2})(\d{2})", month_symbol.lower())
        if not match:
            raise ValueError(f"无法解析合约月份: {month_symbol}")
        yy = int(match.group(1))
        mm = int(match.group(2))
        return 2000 + yy if yy < 80 else 1900 + yy, mm

    @staticmethod
    def _load_trade_calendar() -> set[pd.Timestamp]:
        try:
            cal = ak.tool_trade_date_hist_sina()
            col = "trade_date" if "trade_date" in cal.columns else cal.columns[0]
            return set(pd.to_datetime(cal[col]).dt.normalize())
        except Exception:
            return set()

    @staticmethod
    def _adjust_expiry(expiry: pd.Timestamp, trade_calendar: set[pd.Timestamp]) -> pd.Timestamp:
        expiry = expiry.normalize()
        if not trade_calendar:
            return expiry
        day = expiry
        for _ in range(10):
            if day in trade_calendar:
                return day
            day += pd.Timedelta(days=1)
        return expiry

    def month_expiry(self, month_symbol: str, trade_calendar: set[pd.Timestamp]) -> pd.Timestamp:
        year, month = self._parse_io_month(month_symbol)
        return self._adjust_expiry(self._third_friday(year, month), trade_calendar)

    def _load_rate_points(self, valuation_date: pd.Timestamp) -> list[tuple[int, float]]:
        if self._rate_points is not None:
            return self._rate_points

        points: list[tuple[int, float]] = []
        for indicator, days in {"1月": 30, "3月": 90}.items():
            try:
                df = ak.rate_interbank(
                    market="上海银行同业拆借市场",
                    symbol="Shibor人民币",
                    indicator=indicator,
                )
                date_col = "报告日" if "报告日" in df.columns else "日期"
                rate_col = "利率"
                df[date_col] = pd.to_datetime(df[date_col])
                df = df[df[date_col] <= valuation_date.normalize()].sort_values(date_col)
                if not df.empty:
                    points.append((days, float(df.iloc[-1][rate_col]) / 100.0))
            except Exception:
                continue

        self._rate_points = sorted(points)
        return self._rate_points

    def get_shibor_rate(self, dte: int, valuation_date: pd.Timestamp) -> float:
        points = self._load_rate_points(valuation_date)
        if not points:
            return self.default_rate
        if len(points) == 1:
            return points[0][1]

        x = np.array([p[0] for p in points], dtype=float)
        y = np.array([p[1] for p in points], dtype=float)
        if dte <= x[0]:
            return float(y[0])
        if dte >= x[-1]:
            return float(y[-1])
        return float(np.interp(dte, x, y))

    def fetch_current_chain(self, valuation_time: datetime | None = None) -> pd.DataFrame:
        valuation_ts = self._local_timestamp(valuation_time)
        trade_calendar = self._load_trade_calendar()
        month_dict = ak.option_cffex_hs300_list_sina()

        if isinstance(month_dict, dict):
            months = list(month_dict.values())[0]
        elif isinstance(month_dict, pd.DataFrame):
            months = month_dict.iloc[:, 0].dropna().astype(str).tolist()
        else:
            raise ValueError(f"无法识别月份列表返回格式: {type(month_dict)}")

        rows: list[dict[str, object]] = []
        for month_symbol in months:
            month_symbol = str(month_symbol).lower()
            expiry = self.month_expiry(month_symbol, trade_calendar)
            dte = int((expiry.normalize() - valuation_ts.normalize()).days)
            if dte < self.min_dte:
                continue

            raw = ak.option_cffex_hs300_spot_sina(symbol=month_symbol)
            for _, row in raw.iterrows():
                strike = self._to_float(row.get("行权价"))
                pairs = [
                    (
                        "C",
                        row.get("看涨合约-标识"),
                        row.get("看涨合约-买价"),
                        row.get("看涨合约-卖价"),
                        row.get("看涨合约-最新价"),
                        row.get("看涨合约-买量"),
                        row.get("看涨合约-卖量"),
                        row.get("看涨合约-持仓量"),
                    ),
                    (
                        "P",
                        row.get("看跌合约-标识"),
                        row.get("看跌合约-买价"),
                        row.get("看跌合约-卖价"),
                        row.get("看跌合约-最新价"),
                        row.get("看跌合约-买量"),
                        row.get("看跌合约-卖量"),
                        row.get("看跌合约-持仓量"),
                    ),
                ]
                for cp, contract, bid, ask, last, bid_vol, ask_vol, oi in pairs:
                    rows.append(
                        {
                            "valuation_time": valuation_ts,
                            "month": month_symbol,
                            "expiry": expiry.normalize(),
                            "dte": dte,
                            "cp": cp,
                            "contract": contract,
                            "strike": strike,
                            "bid": self._to_float(bid),
                            "ask": self._to_float(ask),
                            "last": self._to_float(last),
                            "bid_vol": self._to_int(bid_vol),
                            "ask_vol": self._to_int(ask_vol),
                            "open_interest": self._to_int(oi),
                        }
                    )
            time.sleep(0.15)

        chain = pd.DataFrame(rows)
        if chain.empty:
            raise ValueError("没有抓到可用的沪深300股指期权链。")

        chain["bid"] = pd.to_numeric(chain["bid"], errors="coerce")
        chain["ask"] = pd.to_numeric(chain["ask"], errors="coerce")
        chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
        chain["mid"] = (chain["bid"] + chain["ask"]) / 2.0
        chain["valid_quote"] = (
            chain["strike"].notna()
            & (chain["strike"] > 0)
            & chain["bid"].notna()
            & chain["ask"].notna()
            & (chain["bid"] > 0)
            & (chain["ask"] > 0)
            & (chain["ask"] >= chain["bid"])
            & (chain["mid"] > 0)
        )
        return chain

    @staticmethod
    def _pivot_term(term_chain: pd.DataFrame) -> pd.DataFrame:
        fields = ["bid", "ask", "mid", "open_interest", "bid_vol", "ask_vol", "valid_quote"]
        return term_chain.pivot_table(
            index="strike",
            columns="cp",
            values=fields,
            aggfunc="last",
        ).sort_index()

    def compute_term_variance(self, term_chain: pd.DataFrame, valuation_date: pd.Timestamp) -> TermVariance:
        expiry = pd.Timestamp(term_chain["expiry"].iloc[0]).normalize()
        dte = int(term_chain["dte"].iloc[0])
        t_year = dte / 365.0
        rate = self.get_shibor_rate(dte, valuation_date)
        pivot = self._pivot_term(term_chain)

        for col in [("mid", "C"), ("mid", "P"), ("bid", "C"), ("bid", "P")]:
            if col not in pivot.columns:
                raise ValueError(f"{expiry.date()} 缺少字段 {col}")

        valid_pair = (
            pivot[("valid_quote", "C")].fillna(False).astype(bool)
            & pivot[("valid_quote", "P")].fillna(False).astype(bool)
        )
        paired = pivot[valid_pair].copy()
        if len(paired) < 3:
            raise ValueError(f"{expiry.date()} 有效认购认沽配对行权价不足。")

        paired["abs_cp_diff"] = (paired[("mid", "C")] - paired[("mid", "P")]).abs()
        k_star = float(paired["abs_cp_diff"].idxmin())
        c_star = float(paired.loc[k_star, ("mid", "C")])
        p_star = float(paired.loc[k_star, ("mid", "P")])
        forward = k_star + math.exp(rate * t_year) * (c_star - p_star)

        strikes = np.array(sorted(pivot.index.astype(float)))
        below_forward = strikes[strikes <= forward]
        if len(below_forward) == 0:
            raise ValueError(f"{expiry.date()} 无法确定 K0。")
        k0 = float(below_forward[-1])

        selected: list[dict[str, object]] = []
        notes: list[str] = []
        n_puts = 0
        n_calls = 0

        try:
            if bool(pivot.loc[k0, ("valid_quote", "C")]) and bool(pivot.loc[k0, ("valid_quote", "P")]):
                q0 = 0.5 * (float(pivot.loc[k0, ("mid", "C")]) + float(pivot.loc[k0, ("mid", "P")]))
                selected.append({"K": k0, "Q": q0, "side": "K0"})
            else:
                notes.append("K0报价无效")
        except Exception:
            notes.append("K0缺失")

        zero_count = 0
        for k in sorted([k for k in strikes if k < k0], reverse=True):
            bid = self._to_float(pivot.loc[k, ("bid", "P")])
            valid = bool(pivot.loc[k, ("valid_quote", "P")])
            if not valid or bid <= 0:
                if bid <= 0 or pd.isna(bid):
                    zero_count += 1
                if zero_count >= 2:
                    break
                continue
            zero_count = 0
            selected.append({"K": float(k), "Q": float(pivot.loc[k, ("mid", "P")]), "side": "P"})
            n_puts += 1

        zero_count = 0
        for k in sorted([k for k in strikes if k > k0]):
            bid = self._to_float(pivot.loc[k, ("bid", "C")])
            valid = bool(pivot.loc[k, ("valid_quote", "C")])
            if not valid or bid <= 0:
                if bid <= 0 or pd.isna(bid):
                    zero_count += 1
                if zero_count >= 2:
                    break
                continue
            zero_count = 0
            selected.append({"K": float(k), "Q": float(pivot.loc[k, ("mid", "C")]), "side": "C"})
            n_calls += 1

        strip = pd.DataFrame(selected).dropna().sort_values("K").reset_index(drop=True)
        if len(strip) < self.min_term_options:
            raise ValueError(f"{expiry.date()} 纳入期权数量过少: {len(strip)}")

        strikes = strip["K"].to_numpy(dtype=float)
        prices = strip["Q"].to_numpy(dtype=float)
        delta_k = np.empty(len(strikes), dtype=float)
        for i in range(len(strikes)):
            if i == 0:
                delta_k[i] = strikes[i + 1] - strikes[i]
            elif i == len(strikes) - 1:
                delta_k[i] = strikes[i] - strikes[i - 1]
            else:
                delta_k[i] = (strikes[i + 1] - strikes[i - 1]) / 2.0

        contribution = (delta_k / (strikes**2)) * math.exp(rate * t_year) * prices
        variance = (2.0 / t_year) * contribution.sum() - (1.0 / t_year) * ((forward / k0 - 1.0) ** 2)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError(f"{expiry.date()} 方差异常: {variance}")

        quality = "OK"
        if len(strip) < 12 or n_puts < 3 or n_calls < 3:
            quality = "WARN"
            notes.append("尾部期权数量偏少")
        if "K0报价无效" in notes or "K0缺失" in notes:
            quality = "WARN"

        return TermVariance(
            expiry=expiry,
            dte=dte,
            t_year=t_year,
            variance=float(variance),
            forward=float(forward),
            k0=float(k0),
            n_options=int(len(strip)),
            n_puts=int(n_puts),
            n_calls=int(n_calls),
            rate=float(rate),
            quality=quality,
            note="；".join(notes) if notes else "正常",
        )

    def compute_index_from_chain(self, chain: pd.DataFrame) -> dict[str, object]:
        valuation_time = pd.Timestamp(chain["valuation_time"].iloc[0])
        valuation_date = valuation_time.normalize()
        terms = []
        for _, term_chain in chain.groupby("expiry"):
            try:
                terms.append(self.compute_term_variance(term_chain, valuation_date))
            except Exception:
                continue
        if not terms:
            raise ValueError("没有任何有效期限可用于计算。")

        terms = sorted(terms, key=lambda item: item.dte)
        target = self.target_days
        exact = [item for item in terms if item.dte == target]
        notes: list[str] = []
        quality = "OK"

        if exact:
            near = nxt = exact[0]
            var30 = near.variance
            notes.append("存在精确30日到期期限，使用单期限")
        else:
            below = [item for item in terms if item.dte < target]
            above = [item for item in terms if item.dte > target]
            if below and above:
                near = below[-1]
                nxt = above[0]
                notes.append("正常双期限插值")
            else:
                if len(terms) < 2:
                    near = nxt = terms[0]
                    var30 = near.variance
                    quality = "LOW"
                    notes.append("仅一个有效期限，使用单期限近似")
                    return self._build_output(valuation_time, near, nxt, var30, quality, notes)
                chosen = sorted(terms, key=lambda item: abs(item.dte - target))[:2]
                near, nxt = sorted(chosen, key=lambda item: item.dte)
                quality = "WARN"
                notes.append("两个期限未能夹住30日，使用最近期限外推/内插")

            if near.dte == nxt.dte:
                var30 = near.variance
                quality = "WARN"
                notes.append("两个期限DTE相同，使用近端方差")
            else:
                t_target = target / 365.0
                total_var_target = (
                    near.t_year * near.variance * (nxt.dte - target) / (nxt.dte - near.dte)
                    + nxt.t_year * nxt.variance * (target - near.dte) / (nxt.dte - near.dte)
                )
                var30 = total_var_target / t_target
                if var30 <= 0 or not np.isfinite(var30):
                    nearest = sorted(terms, key=lambda item: abs(item.dte - target))[0]
                    near = nxt = nearest
                    var30 = nearest.variance
                    quality = "LOW"
                    notes.append("插值方差异常，回退到最近期限")

        if near.quality != "OK" or nxt.quality != "OK":
            quality = "WARN" if quality == "OK" else quality
            notes.append(f"期限质量: near={near.quality}, next={nxt.quality}")

        return self._build_output(valuation_time, near, nxt, var30, quality, notes)

    @staticmethod
    def _build_output(
        valuation_time: pd.Timestamp,
        near: TermVariance,
        nxt: TermVariance,
        var30: float,
        quality: str,
        notes: list[str],
    ) -> dict[str, object]:
        avix = 100.0 * math.sqrt(max(var30, 0.0))
        return {
            "valuation_time": valuation_time.strftime("%Y-%m-%d %H:%M:%S"),
            "trade_date": valuation_time.strftime("%Y-%m-%d"),
            "avix": round(avix, 4),
            "var30": float(var30),
            "near_expiry": near.expiry.strftime("%Y-%m-%d"),
            "next_expiry": nxt.expiry.strftime("%Y-%m-%d"),
            "near_dte": near.dte,
            "next_dte": nxt.dte,
            "near_var": near.variance,
            "next_var": nxt.variance,
            "near_forward": near.forward,
            "next_forward": nxt.forward,
            "near_k0": near.k0,
            "next_k0": nxt.k0,
            "near_n_options": near.n_options,
            "next_n_options": nxt.n_options,
            "near_rate": near.rate,
            "next_rate": nxt.rate,
            "quality": quality,
            "note": "；".join(notes),
        }


class AVIX300HistClean(AVIX300CloseMid):
    """AVIX-300-HIST-CLEAN: generated historical IO chains from daily close."""

    @staticmethod
    def load_hs300_daily(start_date: str = "20191223", end_date: str = "22220101") -> pd.DataFrame:
        raw = ak.stock_zh_index_daily(symbol="sh000300")
        if raw.empty:
            return pd.DataFrame()
        df = raw.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        df = df.dropna(subset=["date", "close"])
        df = df[(df["date"] >= start) & (df["date"] <= end)].sort_values("date")
        return df[["date", "close"]].reset_index(drop=True)

    @staticmethod
    def _add_month(year: int, month: int, offset: int) -> tuple[int, int]:
        serial = year * 12 + month - 1 + offset
        return serial // 12, serial % 12 + 1

    @staticmethod
    def _quarter_months_after(year: int, month: int, count: int = 3) -> list[tuple[int, int]]:
        out = []
        y, m = year, month
        for _ in range(24):
            y, m = AVIX300HistClean._add_month(y, m, 1)
            if m in {3, 6, 9, 12}:
                out.append((y, m))
                if len(out) >= count:
                    return out
        return out

    @classmethod
    def option_months_for_date(cls, trade_date: pd.Timestamp) -> list[tuple[str, bool]]:
        year = int(trade_date.year)
        month = int(trade_date.month)
        serial_months = [cls._add_month(year, month, offset) for offset in range(3)]
        months: list[tuple[int, int, bool]] = [(y, m, True) for y, m in serial_months]
        for y, m in cls._quarter_months_after(year, month, 3):
            if (y, m) not in [(a, b) for a, b, _ in months]:
                months.append((y, m, False))
        return [(f"io{str(y)[-2:]}{m:02d}", is_serial) for y, m, is_serial in months[:6]]

    @staticmethod
    def strike_step(underlying_close: float, is_serial_month: bool = True) -> int:
        if is_serial_month:
            if underlying_close <= 2500:
                return 25
            if underlying_close <= 5000:
                return 50
            if underlying_close <= 10000:
                return 100
            return 200
        if underlying_close <= 2500:
            return 50
        if underlying_close <= 5000:
            return 100
        if underlying_close <= 10000:
            return 200
        return 400

    @classmethod
    def generate_strikes(cls, prev_close: float, is_serial_month: bool = True) -> list[int]:
        step = cls.strike_step(prev_close, is_serial_month)
        low = math.floor(prev_close * 0.90 / step) * step
        high = math.ceil(prev_close * 1.10 / step) * step
        return list(range(int(low), int(high) + int(step), int(step)))

    @staticmethod
    def _parse_contract(symbol: str) -> tuple[str, str, float]:
        match = re.search(r"^(io\d{4})([CP])(\d+)$", str(symbol).lower(), flags=re.I)
        if not match:
            raise ValueError(f"无法解析期权合约: {symbol}")
        return match.group(1).lower(), match.group(2).upper(), float(match.group(3))

    def candidate_contracts(self, hs300_daily: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for i in range(1, len(hs300_daily)):
            trade_date = pd.Timestamp(hs300_daily.iloc[i]["date"])
            prev_close = float(hs300_daily.iloc[i - 1]["close"])
            for month_symbol, is_serial in self.option_months_for_date(trade_date):
                expiry = self.month_expiry(month_symbol, set())
                if (expiry.normalize() - trade_date.normalize()).days < self.min_dte:
                    continue
                for strike in self.generate_strikes(prev_close, is_serial):
                    for cp in ["C", "P"]:
                        rows.append({"symbol": f"{month_symbol}{cp}{strike}", "month": month_symbol, "cp": cp, "strike": strike})
        return pd.DataFrame(rows).drop_duplicates("symbol").reset_index(drop=True)

    @staticmethod
    def _read_contract_cache() -> pd.DataFrame:
        if not AVIX_CONTRACT_CACHE_FILE.exists():
            return pd.DataFrame()
        try:
            cache = pd.read_csv(AVIX_CONTRACT_CACHE_FILE)
        except Exception:
            return pd.DataFrame()
        if cache.empty:
            return pd.DataFrame()
        cache["date"] = pd.to_datetime(cache["date"], errors="coerce")
        cache["close"] = pd.to_numeric(cache["close"], errors="coerce")
        cache["volume"] = pd.to_numeric(cache["volume"], errors="coerce").fillna(0)
        return cache.dropna(subset=["date", "symbol"])

    @staticmethod
    def _write_contract_cache(cache: pd.DataFrame) -> None:
        if cache.empty:
            return
        out = cache.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out = out.dropna(subset=["date", "symbol"]).drop_duplicates(["symbol", "date"], keep="last")
        out.to_csv(AVIX_CONTRACT_CACHE_FILE, index=False, encoding="utf-8-sig")

    def fetch_contract_daily(self, symbol: str) -> pd.DataFrame:
        try:
            df = ak.option_cffex_hs300_daily_sina(symbol=symbol)
        except Exception:
            return pd.DataFrame()
        if df is None or df.empty:
            return pd.DataFrame()
        out = df.copy()
        out["symbol"] = symbol
        return out[["date", "open", "high", "low", "close", "volume", "symbol"]]

    def build_contract_cache(
        self,
        hs300_daily: pd.DataFrame,
        max_new_contracts: int | None = None,
        sleep_seconds: float = 0.04,
    ) -> pd.DataFrame:
        candidates = self.candidate_contracts(hs300_daily)
        cache = self._read_contract_cache()
        cached_symbols = set(cache["symbol"].astype(str)) if not cache.empty else set()
        missing_symbols = [s for s in candidates["symbol"].astype(str).tolist() if s not in cached_symbols]
        if max_new_contracts is not None:
            missing_symbols = missing_symbols[:max_new_contracts]

        new_parts = []
        for idx, symbol in enumerate(missing_symbols, start=1):
            daily = self.fetch_contract_daily(symbol)
            if not daily.empty:
                new_parts.append(daily)
            if idx % 50 == 0:
                combined = pd.concat([cache] + new_parts, ignore_index=True) if new_parts else cache
                self._write_contract_cache(combined)
            time.sleep(sleep_seconds)

        if new_parts:
            cache = pd.concat([cache] + new_parts, ignore_index=True) if not cache.empty else pd.concat(new_parts, ignore_index=True)
            self._write_contract_cache(cache)
        return self._read_contract_cache()

    def enrich_contract_cache(self, cache: pd.DataFrame) -> pd.DataFrame:
        if cache.empty:
            return cache
        res = cache.copy()
        parsed = res["symbol"].astype(str).apply(self._parse_contract)
        res["month"] = parsed.apply(lambda x: x[0])
        res["cp"] = parsed.apply(lambda x: x[1])
        res["strike"] = parsed.apply(lambda x: x[2])
        res["date"] = pd.to_datetime(res["date"], errors="coerce")
        res["close"] = pd.to_numeric(res["close"], errors="coerce")
        res["volume"] = pd.to_numeric(res["volume"], errors="coerce").fillna(0)
        trade_calendar = self._load_trade_calendar()
        expiry_map = {m: self.month_expiry(m, trade_calendar) for m in res["month"].dropna().unique()}
        res["expiry"] = res["month"].map(expiry_map)
        res["dte"] = (pd.to_datetime(res["expiry"]) - res["date"].dt.normalize()).dt.days
        return res.dropna(subset=["date", "month", "cp", "strike", "expiry"])

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @classmethod
    def black76_price(cls, forward: float, strike: float, t_year: float, rate: float, vol: float, cp: str) -> float:
        if forward <= 0 or strike <= 0 or t_year <= 0 or vol <= 0:
            intrinsic = max(forward - strike, 0) if cp == "C" else max(strike - forward, 0)
            return math.exp(-rate * max(t_year, 0)) * intrinsic
        sigma_t = vol * math.sqrt(t_year)
        d1 = (math.log(forward / strike) + 0.5 * sigma_t * sigma_t) / sigma_t
        d2 = d1 - sigma_t
        disc = math.exp(-rate * t_year)
        if cp == "C":
            return disc * (forward * cls._norm_cdf(d1) - strike * cls._norm_cdf(d2))
        return disc * (strike * cls._norm_cdf(-d2) - forward * cls._norm_cdf(-d1))

    @classmethod
    def implied_vol_black76(cls, price: float, forward: float, strike: float, t_year: float, rate: float, cp: str) -> float:
        if price <= 0 or forward <= 0 or strike <= 0 or t_year <= 0:
            return np.nan
        intrinsic = cls.black76_price(forward, strike, t_year, rate, 1e-6, cp)
        if price < intrinsic * 0.98:
            return np.nan
        lo, hi = 0.01, 1.5
        for _ in range(48):
            mid = (lo + hi) / 2.0
            val = cls.black76_price(forward, strike, t_year, rate, mid, cp)
            if val > price:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    def _estimate_forward(self, pivot: pd.DataFrame, t_year: float, rate: float) -> float:
        paired = pivot.dropna(subset=["C", "P"]).copy()
        paired = paired[(paired["C"] > 0) & (paired["P"] > 0)]
        if len(paired) < 3:
            raise ValueError("配对行权价不足")
        paired["cp_diff"] = (paired["C"] - paired["P"]).abs()
        center = paired.nsmallest(min(9, len(paired)), "cp_diff").copy()
        center["forward_k"] = center.index.astype(float) + math.exp(rate * t_year) * (center["C"] - center["P"])
        if "vol_pair" in center.columns:
            center = center.sort_values("forward_k")
            weights = center["vol_pair"].fillna(1).clip(lower=1).to_numpy(dtype=float)
            csum = weights.cumsum()
            return float(center.iloc[int(np.searchsorted(csum, csum[-1] / 2.0))]["forward_k"])
        return float(center["forward_k"].median())

    def _smooth_term_chain(self, term_chain: pd.DataFrame, valuation_date: pd.Timestamp) -> pd.DataFrame:
        dte = int(term_chain["dte"].iloc[0])
        t_year = dte / 365.0
        rate = self.get_shibor_rate(dte, valuation_date)
        pivot = term_chain.pivot_table(index="strike", columns="cp", values="close", aggfunc="last").sort_index()
        vol_pivot = term_chain.pivot_table(index="strike", columns="cp", values="volume", aggfunc="sum").sort_index()
        if "C" not in pivot.columns or "P" not in pivot.columns:
            return term_chain.assign(clean_price=term_chain["close"])
        paired_vol = vol_pivot.sum(axis=1) if not vol_pivot.empty else pd.Series(1, index=pivot.index)
        pivot["vol_pair"] = paired_vol
        forward = self._estimate_forward(pivot, t_year, rate)

        rows = []
        for _, row in term_chain.iterrows():
            iv = self.implied_vol_black76(float(row["close"]), forward, float(row["strike"]), t_year, rate, str(row["cp"]))
            rows.append(iv)
        clean = term_chain.copy()
        clean["iv"] = rows
        clean["log_moneyness"] = np.log(clean["strike"] / forward)
        clean.loc[(clean["iv"] < 0.01) | (clean["iv"] > 1.5), "iv"] = np.nan

        smooth_parts = []
        for cp, sub in clean.groupby("cp"):
            sub = sub.sort_values("log_moneyness").copy()
            sub["iv_smooth"] = sub["iv"].rolling(5, center=True, min_periods=2).median()
            sub["iv_smooth"] = sub["iv_smooth"].fillna(sub["iv"])
            sub["iv_smooth"] = sub["iv_smooth"].interpolate(limit_direction="both")
            if sub["iv_smooth"].isna().all():
                sub["iv_smooth"] = sub["iv"].median()
            smooth_parts.append(sub)
        clean = pd.concat(smooth_parts, ignore_index=True)
        clean["iv_smooth"] = clean["iv_smooth"].fillna(clean["iv_smooth"].median()).clip(0.01, 1.5)
        clean["clean_price"] = clean.apply(
            lambda r: self.black76_price(forward, float(r["strike"]), t_year, rate, float(r["iv_smooth"]), str(r["cp"])),
            axis=1,
        )
        clean["forward_hint"] = forward
        return clean

    def build_daily_chain(self, cache: pd.DataFrame, trade_date: pd.Timestamp, clean_surface: bool = True) -> pd.DataFrame:
        day = pd.Timestamp(trade_date).normalize()
        chain = cache[cache["date"].dt.normalize() == day].copy()
        chain = chain[(chain["close"] > 0) & (chain["volume"] > 0) & (chain["dte"] >= self.min_dte)].copy()
        if chain.empty:
            return chain
        if clean_surface:
            parts = []
            for _, sub in chain.groupby("expiry"):
                try:
                    parts.append(self._smooth_term_chain(sub, day))
                except Exception:
                    parts.append(sub.assign(clean_price=sub["close"]))
            chain = pd.concat(parts, ignore_index=True)
            chain["price"] = chain["clean_price"]
            chain["source"] = "HIST_CLEAN_SURFACE"
        else:
            chain["price"] = chain["close"]
            chain["source"] = "HIST_RAW_CLOSE"
        return chain

    def compute_hist_term_variance(self, term_chain: pd.DataFrame, valuation_date: pd.Timestamp) -> TermVariance:
        expiry = pd.Timestamp(term_chain["expiry"].iloc[0]).normalize()
        dte = int(term_chain["dte"].iloc[0])
        t_year = dte / 365.0
        rate = self.get_shibor_rate(dte, valuation_date)
        pivot = term_chain.pivot_table(index="strike", columns="cp", values="price", aggfunc="last").sort_index()
        volume_pivot = term_chain.pivot_table(index="strike", columns="cp", values="volume", aggfunc="sum").sort_index()
        if "C" not in pivot.columns or "P" not in pivot.columns:
            raise ValueError(f"{expiry.date()} 缺少C/P")
        pivot["vol_pair"] = volume_pivot.sum(axis=1) if not volume_pivot.empty else 1
        forward = self._estimate_forward(pivot, t_year, rate)
        strikes = np.array(sorted(pivot.index.astype(float)))
        below = strikes[strikes <= forward]
        if len(below) == 0:
            raise ValueError("无法确定K0")
        k0 = float(below[-1])

        selected = []
        n_puts = 0
        n_calls = 0
        if k0 in pivot.index and pd.notna(pivot.loc[k0].get("C")) and pd.notna(pivot.loc[k0].get("P")):
            selected.append({"K": k0, "Q": 0.5 * (float(pivot.loc[k0, "C"]) + float(pivot.loc[k0, "P"]))})

        zero_count = 0
        for k in sorted([k for k in strikes if k < k0], reverse=True):
            q = pivot.loc[k, "P"] if "P" in pivot.columns else np.nan
            vol = volume_pivot.loc[k, "P"] if ("P" in volume_pivot.columns and k in volume_pivot.index) else 0
            if pd.isna(q) or q <= 0 or vol <= 0 or abs(math.log(k / forward)) > 0.40:
                zero_count += 1
                if zero_count >= 2:
                    break
                continue
            zero_count = 0
            selected.append({"K": float(k), "Q": float(q)})
            n_puts += 1

        zero_count = 0
        for k in sorted([k for k in strikes if k > k0]):
            q = pivot.loc[k, "C"] if "C" in pivot.columns else np.nan
            vol = volume_pivot.loc[k, "C"] if ("C" in volume_pivot.columns and k in volume_pivot.index) else 0
            if pd.isna(q) or q <= 0 or vol <= 0 or abs(math.log(k / forward)) > 0.40:
                zero_count += 1
                if zero_count >= 2:
                    break
                continue
            zero_count = 0
            selected.append({"K": float(k), "Q": float(q)})
            n_calls += 1

        strip = pd.DataFrame(selected).dropna().sort_values("K")
        if len(strip) < self.min_term_options:
            raise ValueError(f"纳入期权数量过少: {len(strip)}")
        ks = strip["K"].to_numpy(dtype=float)
        qs = strip["Q"].to_numpy(dtype=float)
        delta_k = np.empty(len(ks), dtype=float)
        for i in range(len(ks)):
            if i == 0:
                delta_k[i] = ks[i + 1] - ks[i]
            elif i == len(ks) - 1:
                delta_k[i] = ks[i] - ks[i - 1]
            else:
                delta_k[i] = (ks[i + 1] - ks[i - 1]) / 2.0
        contribution = (delta_k / (ks**2)) * math.exp(rate * t_year) * qs
        variance = (2.0 / t_year) * contribution.sum() - (1.0 / t_year) * ((forward / k0 - 1.0) ** 2)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError(f"方差异常: {variance}")
        quality = "OK"
        notes = ["历史close链"]
        if len(strip) < 12 or n_puts < 3 or n_calls < 3:
            quality = "WARN"
            notes.append("尾部数量偏少")
        return TermVariance(expiry, dte, t_year, float(variance), float(forward), k0, len(strip), n_puts, n_calls, rate, quality, "；".join(notes))

    def compute_hist_index_from_chain(self, chain: pd.DataFrame, trade_date: pd.Timestamp) -> dict[str, object]:
        terms = []
        for _, sub in chain.groupby("expiry"):
            try:
                terms.append(self.compute_hist_term_variance(sub, pd.Timestamp(trade_date)))
            except Exception:
                continue
        if not terms:
            raise ValueError("没有有效期限")
        terms = sorted(terms, key=lambda x: x.dte)
        target = self.target_days
        below = [x for x in terms if x.dte < target]
        above = [x for x in terms if x.dte > target]
        quality = "OK"
        notes = []
        if below and above:
            near, nxt = below[-1], above[0]
            notes.append("正常双期限插值")
        elif len(terms) >= 2:
            near, nxt = sorted(terms, key=lambda x: abs(x.dte - target))[:2]
            near, nxt = sorted([near, nxt], key=lambda x: x.dte)
            quality = "WARN"
            notes.append("期限未夹住30日")
        else:
            near = nxt = terms[0]
            var30 = near.variance
            quality = "LOW"
            notes.append("单期限")
            return self._build_output(pd.Timestamp(trade_date), near, nxt, var30, quality, notes)
        if near.dte == nxt.dte:
            var30 = near.variance
        else:
            total_var_target = (
                near.t_year * near.variance * (nxt.dte - target) / (nxt.dte - near.dte)
                + nxt.t_year * nxt.variance * (target - near.dte) / (nxt.dte - near.dte)
            )
            var30 = total_var_target / (target / 365.0)
        if near.quality != "OK" or nxt.quality != "OK":
            quality = "WARN" if quality == "OK" else quality
            notes.append(f"期限质量 near={near.quality}, next={nxt.quality}")
        return self._build_output(pd.Timestamp(trade_date), near, nxt, var30, quality, notes)

    def backfill_hist_clean(
        self,
        start_date: str = "20191223",
        end_date: str = "22220101",
        max_new_contracts: int | None = None,
        clean_surface: bool = True,
    ) -> pd.DataFrame:
        hs300 = self.load_hs300_daily(start_date=start_date, end_date=end_date)
        if hs300.empty:
            return pd.DataFrame()
        cache = self.build_contract_cache(hs300, max_new_contracts=max_new_contracts)
        cache = self.enrich_contract_cache(cache)
        rows = []
        for trade_date in hs300["date"].iloc[1:]:
            chain = self.build_daily_chain(cache, pd.Timestamp(trade_date), clean_surface=clean_surface)
            if chain.empty:
                continue
            try:
                result = self.compute_hist_index_from_chain(chain, pd.Timestamp(trade_date))
            except Exception:
                continue
            result["source"] = "HIST_CLEAN_SURFACE" if clean_surface else "HIST_RAW_CLOSE"
            rows.append(result)
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        out["trade_date"] = pd.to_datetime(out["trade_date"])
        out = out.sort_values("trade_date").drop_duplicates("trade_date", keep="last")
        path = AVIX_HIST_FILE if clean_surface else AVIX_RAW_CLOSE_FILE
        out.to_csv(path, index=False, encoding="utf-8-sig")
        return out


def _append_dedup_csv(path: Path, row_df: pd.DataFrame, subset: list[str]) -> None:
    if path.exists():
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()
        out = pd.concat([old, row_df], ignore_index=True)
    else:
        out = row_df
    out = out.drop_duplicates(subset=subset, keep="last")
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _filter_to_trading_days(df: pd.DataFrame, date_col: str = "trade_date") -> pd.DataFrame:
    if df.empty or date_col not in df.columns:
        return df
    out = df.copy()
    trade_dates = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    calendar = AVIX300CloseMid._load_trade_calendar()
    if calendar:
        mask = trade_dates.isin(calendar)
    else:
        mask = trade_dates.dt.weekday < 5
    return out[mask.fillna(False)].copy()


def calculate_and_store_avix() -> dict[str, object]:
    model = AVIX300CloseMid()
    chain = model.fetch_current_chain()
    result = model.compute_index_from_chain(chain)
    result = {**result, "source": "CLOSE_MID"}
    if not _filter_to_trading_days(pd.DataFrame([result])).empty:
        _append_dedup_csv(AVIX_INDEX_FILE, pd.DataFrame([result]), ["trade_date"])

        chain_out = chain.copy()
        chain_out["valuation_day"] = pd.to_datetime(chain_out["valuation_time"]).dt.strftime("%Y-%m-%d")
        _append_dedup_csv(AVIX_CHAIN_FILE, chain_out, ["valuation_day", "contract"])
    else:
        result["note"] = f'{result.get("note", "")}；非交易日未写入历史'.strip("；")
    return result


def backfill_avix_history_qvix() -> pd.DataFrame:
    """Backfill historical HS300 index-option volatility from AKShare QVIX.

    Missing 300-index QVIX days are filled one by one from same-day 300ETF QVIX
    with a rolling median ratio calibration on overlapping history.
    """
    try:
        raw = ak.index_option_300index_qvix()
    except Exception:
        return pd.DataFrame()

    if raw.empty or "date" not in raw.columns or "close" not in raw.columns:
        return pd.DataFrame()

    hist = raw.copy()
    hist["trade_date"] = pd.to_datetime(hist["date"], errors="coerce")
    hist["index_qvix"] = pd.to_numeric(hist["close"], errors="coerce")
    hist = hist.dropna(subset=["trade_date"]).copy()
    hist = hist[hist["trade_date"] >= pd.Timestamp("2019-12-23")].copy()
    hist["source"] = "HIST_QVIX"
    hist["note"] = "AKShare index_option_300index_qvix 历史序列"

    try:
        etf = ak.index_option_300etf_qvix()
        etf["trade_date"] = pd.to_datetime(etf["date"], errors="coerce")
        etf["etf_qvix"] = pd.to_numeric(etf["close"], errors="coerce")
        etf = etf.dropna(subset=["trade_date"])[["trade_date", "etf_qvix"]]
        hist = hist.merge(etf, on="trade_date", how="left")
    except Exception:
        hist["etf_qvix"] = np.nan

    overlap = hist[(hist["index_qvix"] > 0) & (hist["etf_qvix"] > 0)].copy()
    overlap["ratio"] = overlap["index_qvix"] / overlap["etf_qvix"]
    default_ratio = float(overlap["ratio"].median()) if not overlap.empty else 1.0

    hist["avix"] = hist["index_qvix"]
    missing_mask = hist["avix"].isna() | (hist["avix"] <= 0)
    for idx in hist.index[missing_mask]:
        trade_date = hist.at[idx, "trade_date"]
        etf_value = hist.at[idx, "etf_qvix"]
        if pd.notna(etf_value) and etf_value > 0:
            prior = overlap[overlap["trade_date"] < trade_date].tail(252)
            ratio = float(prior["ratio"].median()) if not prior.empty else default_ratio
            hist.at[idx, "avix"] = float(etf_value) * ratio
            hist.at[idx, "source"] = "HIST_PROXY_300ETF"
            hist.at[idx, "note"] = f"300index QVIX缺失；用同日300ETF QVIX按历史中位比例{ratio:.4f}校准补值"

    hist["avix"] = pd.to_numeric(hist["avix"], errors="coerce")
    hist = hist.dropna(subset=["avix"]).copy()
    hist = hist[hist["avix"] > 0].copy()
    if hist.empty:
        return pd.DataFrame()

    hist["valuation_time"] = hist["trade_date"].dt.strftime("%Y-%m-%d 15:00:00")
    hist["var30"] = (hist["avix"] / 100.0) ** 2
    hist["near_expiry"] = ""
    hist["next_expiry"] = ""
    hist["near_dte"] = np.nan
    hist["next_dte"] = np.nan
    hist["near_var"] = np.nan
    hist["next_var"] = np.nan
    hist["near_forward"] = np.nan
    hist["next_forward"] = np.nan
    hist["near_k0"] = np.nan
    hist["next_k0"] = np.nan
    hist["near_n_options"] = np.nan
    hist["next_n_options"] = np.nan
    hist["near_rate"] = np.nan
    hist["next_rate"] = np.nan
    hist["quality"] = "HIST"

    out_cols = [
        "valuation_time", "trade_date", "avix", "var30", "near_expiry", "next_expiry",
        "near_dte", "next_dte", "near_var", "next_var", "near_forward", "next_forward",
        "near_k0", "next_k0", "near_n_options", "next_n_options", "near_rate",
        "next_rate", "quality", "note", "source",
    ]
    out = hist[out_cols].sort_values("trade_date").reset_index(drop=True)
    out.to_csv(AVIX_QVIX_FALLBACK_FILE, index=False, encoding="utf-8-sig")
    return out


def load_avix_history() -> pd.DataFrame:
    frames = []
    history_paths = [AVIX_HIST_FILE, AVIX_INDEX_FILE]
    if not AVIX_HIST_FILE.exists():
        history_paths.insert(0, AVIX_QVIX_FALLBACK_FILE)

    for path in history_paths:
        if not path.exists():
            continue
        try:
            part = pd.read_csv(path)
        except Exception:
            continue
        if not part.empty:
            if path == AVIX_INDEX_FILE:
                part["source"] = "CLOSE_MID"
                part = _filter_to_trading_days(part)
            elif path == AVIX_QVIX_FALLBACK_FILE and "source" not in part.columns:
                part["source"] = "HIST_QVIX"
            elif "source" not in part.columns:
                part["source"] = "HIST_CLEAN_SURFACE"
            frames.append(part)

    if not frames:
        return pd.DataFrame()

    hist = pd.concat(frames, ignore_index=True)
    if hist.empty or "trade_date" not in hist.columns or "avix" not in hist.columns:
        return pd.DataFrame()
    hist = hist.copy()
    hist["trade_date"] = pd.to_datetime(hist["trade_date"], errors="coerce")
    hist["avix"] = pd.to_numeric(hist["avix"], errors="coerce")
    hist = hist.dropna(subset=["trade_date", "avix"]).sort_values("trade_date")
    if "source" not in hist.columns:
        hist["source"] = "UNKNOWN"
    hist["_priority"] = hist["source"].map({
        "HIST_QVIX": 0,
        "HIST_PROXY_300ETF": 0,
        "HIST_RAW_CLOSE": 1,
        "HIST_CLEAN_SURFACE": 2,
        "CLOSE_MID": 3,
    }).fillna(0)
    hist = (
        hist.sort_values(["trade_date", "_priority"])
        .drop_duplicates("trade_date", keep="last")
        .drop(columns=["_priority"])
    )
    return hist.reset_index(drop=True)
