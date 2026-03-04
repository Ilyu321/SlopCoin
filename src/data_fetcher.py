import os
import json
import time
import logging
import ccxt
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
from typing import Optional, Dict, List, Tuple, Any
from config import (
    BASE_CURRENCY, KRAKEN_API_PATH, CCXT_TIMEOUT_SECONDS,
    PRICE_CACHE_TTL, PRICE_CACHE_TTL_STATIC, PRICE_CACHE_TTL_MIN, PRICE_CACHE_TTL_MAX,
    VOLATILITY_LOOKBACK, MAX_HISTORY_PER_COIN, MAX_TOTAL_HISTORY_ENTRIES,
    INDICATOR_CACHE_TTL, PORTFOLIO_CACHE_TTL, MARKET_OVERVIEW_TOP_N,
)
from cache_manager import IntelligentCache
from retry import retry

logger = logging.getLogger(__name__)

# Intelligenter Cache (Singleton auf Modul-Ebene)
cache_manager = IntelligentCache(cache_dir="/tmp/cache")


class MarketData:
    """Marktdatenabrufe von Kraken Exchange via CCXT."""

    def __init__(self, secrets_path: str = None):
        """Initialisiert MarketData mit Kraken API Verbindung.

        Args:
            secrets_path: Pfad zur Kraken API JSON-Datei.
                          Default: KRAKEN_API_PATH aus config
        """
        if secrets_path is None:
            secrets_path = KRAKEN_API_PATH

        try:
            with open(secrets_path) as f:
                creds = json.load(f)
        except FileNotFoundError:
            logger.critical(f"Kraken API Credentials nicht gefunden: {secrets_path}")
            raise

        # Read-Only Verbindung zu Kraken
        self.exchange = ccxt.kraken({
            'apiKey': creds['key'],
            'secret': creds['secret'],
            'enableRateLimit': True,
        })
        # Globalen Timeout für alle Requests setzen (ms)
        try:
            self.exchange.timeout = CCXT_TIMEOUT_SECONDS * 1000
        except Exception as e:
            logger.warning(f"Konnte ccxt Timeout nicht setzen: {e}")
        logger.info(f"ccxt Version: {ccxt.__version__}, Timeout: {getattr(self.exchange, 'timeout', 'unknown')} ms")

        # Markets einmalig laden für Verfügbarkeitsprüfung (mit Retry)
        try:
            self._load_markets_with_retry()
        except Exception as e:
            logger.error(f"Fehler beim Laden der Markets nach Retry: {e}")
            self.markets = {}

        # Preis-Historie für Volatilitätsberechnung (in-memory)
        self._price_history: Dict[str, List[float]] = {}

    # ── Interne Hilfsmethoden ────────────────────────────────────────────────

    def _update_price_history(self, coin: str, price: float, max_length: int = VOLATILITY_LOOKBACK) -> None:
        """Aktualisiert die Preis-Historie für Volatilitätsberechnung.

        Args:
            coin: Coin-Symbol
            price: Aktueller Preis
            max_length: Maximale Anzahl zu speichernder Preis-Punkte
        """
        if coin not in self._price_history:
            self._price_history[coin] = []
        self._price_history[coin].append(price)
        self._price_history[coin] = self._price_history[coin][-max_length:]

    def calculate_volatility(self, coin: str) -> float:
        """Berechnet die Volatilität als Standardabweichung der Returns.

        Args:
            coin: Coin-Symbol

        Returns:
            Volatilität als float (0.0 wenn nicht genug Daten)
        """
        history = self._price_history.get(coin, [])
        if len(history) < 2:
            return 0.0
        returns = np.diff(history) / history[:-1]
        if len(returns) < 2:
            return 0.0
        return float(np.std(returns))

    def get_adaptive_ttl(self, coin: str, base_ttl: int) -> int:
        """Berechnet adaptive TTL basierend auf Marktvolatilität.

        Args:
            coin: Coin-Symbol
            base_ttl: Basis-TTL in Sekunden

        Returns:
            Angepasste TTL in Sekunden
        """
        volatility = self.calculate_volatility(coin)
        if volatility > 0.05:   # >5% → kürzere TTL
            return max(PRICE_CACHE_TTL_MIN, int(base_ttl * 0.5))
        elif volatility < 0.01:  # <1% → längere TTL
            return min(PRICE_CACHE_TTL_MAX, int(base_ttl * 2))
        return base_ttl

    def _normalize_symbol(self, coin: str, base_currency: str = BASE_CURRENCY) -> str:
        """Konvertiert Coin-Namen zu Kraken Trading-Paar.

        Args:
            coin: Coin-Symbol (z.B. "BTC")
            base_currency: Basis-Währung (z.B. "EUR")

        Returns:
            Normalisiertes Symbol (z.B. "BTC/EUR")
        """
        candidates = [
            f"{coin}/{base_currency}",
            f"{coin}Z{base_currency}",  # Kraken-Format für einige Coins
        ]
        for symbol in candidates:
            if symbol in self.markets:
                return symbol
        return f"{coin}/{base_currency}"

    # ── Retry-geschützte API-Aufrufe ─────────────────────────────────────────

    @retry(max_attempts=3, base_delay=2.0, max_delay=30.0,
           exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError))
    def _load_markets_with_retry(self) -> None:
        """Lädt Markets mit Retry-Logik."""
        self.markets = self.exchange.load_markets()
        logger.info(f"Markets geladen: {len(self.markets)} verfügbar")

    @retry(max_attempts=3, base_delay=1.0, max_delay=30.0,
           exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError))
    def _fetch_balance_with_retry(self) -> Dict:
        """Holt Kontostand mit Retry-Logik."""
        return self.exchange.fetch_balance()

    @retry(max_attempts=3, base_delay=1.0, max_delay=30.0,
           exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError))
    def _fetch_ticker_with_retry(self, symbol: str) -> Dict:
        """Holt einzelnen Ticker mit Retry-Logik."""
        return self.exchange.fetch_ticker(symbol)

    @retry(max_attempts=3, base_delay=1.0, max_delay=30.0,
           exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError))
    def _fetch_tickers_with_retry(self, symbols: List[str]) -> Dict:
        """Holt mehrere Ticker in einem Batch-API-Call mit Retry-Logik.

        Args:
            symbols: Liste von Trading-Paar-Symbolen

        Returns:
            Dict mit Ticker-Daten pro Symbol
        """
        tickers = self.exchange.fetch_tickers(symbols=symbols)
        logger.debug(f"Batch-Tickers geladen: {len(tickers)} Coins")
        return tickers

    @retry(max_attempts=3, base_delay=1.0, max_delay=30.0,
           exceptions=(ccxt.NetworkError, ccxt.ExchangeError, ConnectionError, TimeoutError))
    def _fetch_ohlcv_with_retry(self, symbol: str, timeframe: str, limit: int = 200) -> List:
        """Holt OHLCV-Daten mit Retry-Logik.

        Args:
            symbol: Trading-Paar-Symbol
            timeframe: Zeitrahmen (z.B. '4h')
            limit: Maximale Anzahl Candles

        Returns:
            Liste von OHLCV-Datenpunkten
        """
        return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

    # ── Öffentliche Datenabruf-Methoden ──────────────────────────────────────

    def get_portfolio(self) -> Dict[str, float]:
        """Holt Kontostand-Positionen mit Betrag > 0.001.

        Returns:
            Dict mit Coin-Symbol → Menge
        """
        cached = cache_manager.get('portfolio')
        if cached is not None:
            logger.debug("Portfolio aus Cache geladen")
            return cached

        try:
            bal = self._fetch_balance_with_retry()
            # Debug-Log der rohen Balance-Struktur (gekürzt), um Strukturprobleme zu erkennen
            try:
                # Nur die wichtigsten Keys loggen, um Log-Spam zu vermeiden
                bal_preview = {k: {sk: sv for sk, sv in v.items() if sk in ['total', 'free', 'used']} for k, v in bal.items() if isinstance(v, dict)}
                logger.debug(f"Raw balance preview: {json.dumps(bal_preview)[:2000]}")
            except Exception as log_e:
                logger.debug(f"Konnte Balance-Preview nicht loggen: {log_e}")

            portfolio: Dict[str, float] = {}
            for k, v in bal.items():
                # ccxt-Balance-Einträge sind i.d.R. Dicts mit Keys wie 'total', 'free', 'used'
                if not isinstance(v, dict):
                    continue
                total = v.get('total')
                # Manche ccxt-Versionen liefern 0 statt None, wir filtern nur wirklich relevante Positionen
                if total is None:
                    continue
                try:
                    total_val = float(total)
                except (TypeError, ValueError):
                    continue
                if total_val > 0.001:
                    portfolio[k] = total_val

            # Debug-Log des normalisierten Portfolios, um Mapping-Probleme (z.B. BTC vs XBT) zu erkennen
            try:
                logger.info(f"Portfolio geladen: {len(portfolio)} Coins -> {json.dumps(portfolio)}")
            except Exception:
                logger.info(f"Portfolio geladen: {len(portfolio)} Coins (Details nicht serialisierbar)")

            cache_manager.set('portfolio', portfolio, ttl=PORTFOLIO_CACHE_TTL)
            return portfolio
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Portfolios: {e}")
            # Fallback: Leeres Portfolio zurückgeben, aber mit Warnung
            logger.warning("Fehler beim Abrufen des Portfolios - verwende leeres Portfolio")
            return {}

    def get_portfolio_with_prices(self) -> Tuple[Dict[str, float], Dict[str, Optional[float]]]:
        """Holt Portfolio + aktuelle Preise pro Coin (optimiert mit Batch-Abruf).

        Returns:
            Tuple aus (portfolio_dict, prices_dict)
        """
        portfolio = self.get_portfolio()
        prices: Dict[str, Optional[float]] = {}

        # Zuerst alle Preise aus Cache laden
        coins_to_fetch: List[Tuple[str, str]] = []
        for coin in portfolio:
            symbol = self._normalize_symbol(coin)
            price_key = f'price_{symbol}'
            cached_price = cache_manager.get(price_key)
            if cached_price is not None:
                prices[coin] = cached_price
            else:
                coins_to_fetch.append((coin, symbol))

        # Batch-Abruf für alle fehlenden Preise
        if coins_to_fetch:
            try:
                symbols = [symbol for _, symbol in coins_to_fetch]
                tickers = self._fetch_tickers_with_retry(symbols)

                for coin, symbol in coins_to_fetch:
                    if symbol in tickers:
                        price = tickers[symbol]['last']
                        prices[coin] = price
                        adaptive_ttl = self.get_adaptive_ttl(coin, PRICE_CACHE_TTL_STATIC)
                        cache_manager.set(f'price_{symbol}', price, ttl=adaptive_ttl)
                        self._update_price_history(coin, price)
                    else:
                        prices[coin] = None
                        logger.warning(f"Kein Preis für {symbol} in Batch-Response")

            except Exception as e:
                logger.error(f"Fehler beim Batch-Preisabruf: {e}")
                # Fallback: Einzelabfragen für alle fehlenden Coins
                for coin, symbol in coins_to_fetch:
                    try:
                        ticker = self._fetch_ticker_with_retry(symbol)
                        price = ticker['last']
                        prices[coin] = price
                        adaptive_ttl = self.get_adaptive_ttl(coin, PRICE_CACHE_TTL_STATIC)
                        cache_manager.set(f'price_{symbol}', price, ttl=adaptive_ttl)
                        self._update_price_history(coin, price)
                    except Exception as e2:
                        logger.warning(f"Preis für {coin} nicht verfügbar: {e2}")
                        prices[coin] = None

        # Basis-Currency (EUR) direkt aus Portfolio übernehmen
        if BASE_CURRENCY in portfolio:
            prices[BASE_CURRENCY] = portfolio[BASE_CURRENCY]

        # Validierung: Alle Portfolio-Coins sollten einen Preis haben
        missing_prices = [coin for coin in portfolio if coin != BASE_CURRENCY and prices.get(coin) is None]
        if missing_prices:
            logger.warning(f"Keine Preise für folgende Coins verfügbar: {missing_prices}")

        return portfolio, prices

    # ── Technische Analyse ────────────────────────────────────────────────────

    def detect_rsi_divergence(self, prices: List[float], rsi_values: List[float], window: int = 14) -> Optional[Dict[str, bool]]:
        """Erkennt Bullish/Bearish RSI-Divergenzen.

        Args:
            prices: Liste von Schlusskursen
            rsi_values: Liste von RSI-Werten
            window: Lookback-Fenster für Pivot-Erkennung

        Returns:
            Dict mit 'bullish' und 'bearish' Flags, oder None bei Fehler
        """
        if len(prices) < window * 2 or len(rsi_values) < window * 2:
            return None

        try:
            recent_prices = prices[-window * 2:]
            recent_rsi = rsi_values[-window * 2:]

            price_highs: List[Tuple[int, float]] = []
            price_lows: List[Tuple[int, float]] = []
            rsi_highs: List[Tuple[int, float]] = []
            rsi_lows: List[Tuple[int, float]] = []

            for i in range(1, len(recent_prices) - 1):
                if recent_prices[i] > recent_prices[i - 1] and recent_prices[i] > recent_prices[i + 1]:
                    price_highs.append((i, recent_prices[i]))
                if recent_prices[i] < recent_prices[i - 1] and recent_prices[i] < recent_prices[i + 1]:
                    price_lows.append((i, recent_prices[i]))
                if recent_rsi[i] > recent_rsi[i - 1] and recent_rsi[i] > recent_rsi[i + 1]:
                    rsi_highs.append((i, recent_rsi[i]))
                if recent_rsi[i] < recent_rsi[i - 1] and recent_rsi[i] < recent_rsi[i + 1]:
                    rsi_lows.append((i, recent_rsi[i]))

            # Bullish Divergence: Preis macht niedrigere Tiefs, RSI macht höhere Tiefs
            bullish_divergence = (
                len(price_lows) >= 2 and len(rsi_lows) >= 2
                and price_lows[-1][1] < price_lows[-2][1]
                and rsi_lows[-1][1] > rsi_lows[-2][1]
            )

            # Bearish Divergence: Preis macht höhere Hochs, RSI macht niedrigere Hochs
            bearish_divergence = (
                len(price_highs) >= 2 and len(rsi_highs) >= 2
                and price_highs[-1][1] > price_highs[-2][1]
                and rsi_highs[-1][1] < rsi_highs[-2][1]
            )

            return {'bullish': bullish_divergence, 'bearish': bearish_divergence}

        except Exception as e:
            logger.warning(f"RSI-Divergenz-Erkennung fehlgeschlagen: {e}")
            return None

    def get_indicators(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Berechnet erweiterte Indikatoren: RSI, SMA200, MACD, Bollinger Bands, OBV, Ichimoku, RSI-Divergenz, Volatilität.

        Args:
            symbol: Trading-Paar-Symbol (z.B. "BTC/EUR")

        Returns:
            Dict mit Indikator-Werten oder None bei Fehler
        """
        cache_key = f'indicators_{symbol}'
        cached = cache_manager.get(cache_key)
        if cached is not None:
            logger.debug(f"Indikatoren für {symbol} aus Cache geladen")
            return cached

        try:
            ohlcv = self._fetch_ohlcv_with_retry(symbol, '4h', limit=200)
            df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])

            if len(df) < 50:
                logger.warning(f"Nicht genug Daten für {symbol}: {len(df)} Candles")
                return None

            # Basis-Indikatoren
            rsi = df.ta.rsi(length=14)
            sma200 = df.ta.sma(length=200)
            macd_data = df.ta.macd(fast=12, slow=26, signal=9)
            bb_data = df.ta.bbands(length=20, std=2)

            # Volatilität (annualisiert in %)
            df['returns'] = df['close'].pct_change()
            volatility_30d = df['returns'].tail(30).std() * np.sqrt(365 * 24 / 4) * 100

            current_price = df['close'].iloc[-1]
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(20).mean()

            # MACD-Signale
            macd_line = macd_data['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in macd_data.columns else None
            macd_signal = macd_data['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in macd_data.columns else None
            macd_histogram = macd_data['MACDh_12_26_9'].iloc[-1] if 'MACDh_12_26_9' in macd_data.columns else None

            macd_bullish = False
            macd_bearish = False
            if macd_line is not None and macd_signal is not None and macd_histogram is not None:
                macd_bullish = macd_line > macd_signal and macd_histogram > 0
                macd_bearish = macd_line < macd_signal and macd_histogram < 0

            # Bollinger Bands
            bb_upper = bb_data['BBU_20_2.0'].iloc[-1] if 'BBU_20_2.0' in bb_data.columns else None
            bb_middle = bb_data['BBM_20_2.0'].iloc[-1] if 'BBM_20_2.0' in bb_data.columns else None
            bb_lower = bb_data['BBL_20_2.0'].iloc[-1] if 'BBL_20_2.0' in bb_data.columns else None

            bb_position: Optional[str] = None
            if bb_upper is not None and bb_lower is not None:
                if current_price >= bb_upper:
                    bb_position = "overbought"
                elif current_price <= bb_lower:
                    bb_position = "oversold"
                else:
                    bb_position = "neutral"

            # Volume-Analyse
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

            # On-Balance Volume (OBV)
            df['obv'] = df.ta.obv()
            current_obv = df['obv'].iloc[-1]
            obv_20ma = df['obv'].rolling(20).mean().iloc[-1]
            obv_trend = "bullish" if current_obv > obv_20ma else "bearish"

            # Ichimoku Cloud
            tenkan = kijun = senkou_a = senkou_b = chikou = None
            cloud_position: Optional[str] = None
            try:
                # pandas-ta-classic gibt direkt einen DataFrame zurück (kein Tuple)
                ichimoku = df.ta.ichimoku()
                tenkan = ichimoku['ITS_9'].iloc[-1] if 'ITS_9' in ichimoku.columns else None
                kijun = ichimoku['IKS_26'].iloc[-1] if 'IKS_26' in ichimoku.columns else None
                senkou_a = ichimoku['ISA_9'].iloc[-1] if 'ISA_9' in ichimoku.columns else None
                senkou_b = ichimoku['ISB_26'].iloc[-1] if 'ISB_26' in ichimoku.columns else None
                chikou = ichimoku['ICS_26'].iloc[-1] if 'ICS_26' in ichimoku.columns else None

                if senkou_a is not None and senkou_b is not None:
                    if current_price > max(senkou_a, senkou_b):
                        cloud_position = "above"
                    elif current_price < min(senkou_a, senkou_b):
                        cloud_position = "below"
                    else:
                        cloud_position = "inside"
            except Exception as e:
                logger.debug(f"Ichimoku-Berechnung fehlgeschlagen: {e}")

            # RSI-Divergenz
            rsi_divergence = self.detect_rsi_divergence(
                df['close'].tolist(),
                rsi.tolist() if not rsi.empty else [],
                window=14,
            )

            result = {
                "symbol": symbol,
                "price": round(float(current_price), 2),
                "rsi_14": round(float(rsi.iloc[-1]), 2) if not rsi.empty else None,
                "sma200": round(float(sma200.iloc[-1]), 2) if not sma200.empty else None,
                "trend": ("bullish" if current_price > sma200.iloc[-1] else "bearish") if not sma200.empty else None,
                "macd_line": round(float(macd_line), 4) if macd_line is not None else None,
                "macd_signal": round(float(macd_signal), 4) if macd_signal is not None else None,
                "macd_histogram": round(float(macd_histogram), 4) if macd_histogram is not None else None,
                "macd_bullish": macd_bullish,
                "macd_bearish": macd_bearish,
                "bb_upper": round(float(bb_upper), 2) if bb_upper is not None else None,
                "bb_middle": round(float(bb_middle), 2) if bb_middle is not None else None,
                "bb_lower": round(float(bb_lower), 2) if bb_lower is not None else None,
                "bb_position": bb_position,
                "volatility_30d": round(float(volatility_30d), 2),
                "volume_ratio": round(float(volume_ratio), 2),
                "current_volume": round(float(current_volume), 2),
                "obv": round(float(current_obv), 0),
                "obv_trend": obv_trend,
                "ichimoku_tenkan": round(float(tenkan), 2) if tenkan is not None else None,
                "ichimoku_kijun": round(float(kijun), 2) if kijun is not None else None,
                "ichimoku_senkou_a": round(float(senkou_a), 2) if senkou_a is not None else None,
                "ichimoku_senkou_b": round(float(senkou_b), 2) if senkou_b is not None else None,
                "ichimoku_cloud_position": cloud_position,
                "rsi_divergence": rsi_divergence,
            }

            cache_manager.set(cache_key, result, ttl=INDICATOR_CACHE_TTL)
            return result

        except Exception as e:
            logger.error(f"Fehler beim Berechnen der Indikatoren für {symbol}: {e}")
            # Fallback: Leere Indikatoren zurückgeben
            logger.warning(f"Fehler bei Indikatoren für {symbol} - verwende leere Indikatoren")
            return {}

    def get_portfolio_indicators(self, portfolio_coins: Dict[str, float]) -> Dict[str, Optional[Dict]]:
        """Berechnet Indikatoren für alle Coins im Portfolio.

        Args:
            portfolio_coins: Dict mit Coin-Symbol → Menge

        Returns:
            Dict mit Coin-Symbol → Indikator-Dict (oder None bei Fehler)
        """
        indicators: Dict[str, Optional[Dict]] = {}
        for coin in portfolio_coins:
            try:
                symbol = self._normalize_symbol(coin)
                indicators[coin] = self.get_indicators(symbol)
                if indicators[coin] is None:
                    logger.warning(f"Indikatoren für {coin} konnten nicht berechnet werden")
            except Exception as e:
                logger.error(f"Fehler bei {coin}: {e}")
                indicators[coin] = None
        return indicators

    def get_market_overview(
        self,
        top_n: Optional[int] = None,
        base_currency: str = BASE_CURRENCY,
        exclude_coins: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """Analysiert die Top-N Markt-Coins (nur Kraken-handelbare).

        Args:
            top_n: Anzahl der zu analysierenden Top-Coins (Default: MARKET_OVERVIEW_TOP_N)
            base_currency: Basis-Währung (z.B. "EUR")
            exclude_coins: Coins die ausgeschlossen werden sollen (z.B. Portfolio-Coins)

        Returns:
            Dict mit Coin-Symbol → Indikator-Dict
        """
        if top_n is None:
            top_n = MARKET_OVERVIEW_TOP_N
        if exclude_coins is None:
            exclude_coins = []

        try:
            # Alle aktiven EUR-Markets finden
            eur_markets = [
                {'symbol': symbol, 'base': market['base'], 'market': market}
                for symbol, market in self.markets.items()
                if market['quote'] == base_currency and market['active']
                and market['base'] not in exclude_coins
            ]

            # Ticker für Top-N*2 Markets abrufen (für bessere Volume-Sortierung)
            tickers = []
            for market_info in eur_markets[:top_n * 2]:
                try:
                    ticker = self._fetch_ticker_with_retry(market_info['symbol'])
                    tickers.append({
                        'symbol': market_info['symbol'],
                        'base': market_info['base'],
                        'volume_24h': ticker.get('quoteVolume', 0) or 0,
                    })
                except Exception as e:
                    logger.debug(f"Ticker für {market_info['symbol']} nicht verfügbar: {e}")

            # Nach Volume sortieren und Top-N auswählen
            tickers.sort(key=lambda x: x['volume_24h'], reverse=True)
            top_markets = tickers[:top_n]

            # Indikatoren für Top-Markets berechnen
            market_overview: Dict[str, Dict] = {}
            for market_info in top_markets:
                coin = market_info['base']
                symbol = market_info['symbol']
                try:
                    indicators = self.get_indicators(symbol)
                    if indicators:
                        market_overview[coin] = indicators
                except Exception as e:
                    logger.warning(f"Indikatoren für {coin} ({symbol}) konnten nicht berechnet werden: {e}")

            logger.info(f"Markt-Übersicht erstellt: {len(market_overview)} Coins analysiert")
            return market_overview

        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Markt-Übersicht: {e}")
            return {}
