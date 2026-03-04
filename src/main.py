import os
import json
import logging
import time
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from llm_engine import LLMEngine
from data_fetcher import MarketData
from portfolio_tracker import PortfolioTracker
from risk_analyzer import RiskAnalyzer
from config import (
    TELEGRAM_TOKEN_PATH,
    ALLOWED_TELEGRAM_USER_ID,
    SCHEDULE_INTERVAL_SECONDS,
    ANALYSIS_START_HOUR,
    ANALYSIS_END_HOUR,
    PAUSE_STATE_PATH,
    PRICE_CHECK_INTERVAL_SECONDS,
    PRICE_ALERT_THRESHOLD_UP,
    PRICE_ALERT_THRESHOLD_DOWN,
    WEEKLY_SUMMARY_DAY,
    WEEKLY_SUMMARY_HOUR,
)
from config_validator import ConfigValidator
from signal_handler import setup_signal_handlers, register_cleanup_function, perform_graceful_shutdown
from input_validator import validate_what_if_args, validate_set_interval_args, validate_next_invest_args

# Logging Setup mit structlog (falls verfügbar)
try:
    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logger = structlog.get_logger()
    logger = logger.bind(component='SlopCoin')
    logger.info("Structured logging initialized")
except ImportError:
    # Fallback auf Standard-Logging
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger("SlopCoin")
    logger.info("Using standard logging (structlog not available)")

# Globals (werden in main() gesetzt)
market = None
tracker = None
risk_analyzer = None
brain = None
ADMIN_ID = None
alert_manager = None
start_time = None
health_server = None


def is_paused() -> bool:
    """Liest Pause-Zustand aus Datei.
    
    Returns:
        bool: True wenn pausiert, False sonst
    """
    try:
        if os.path.exists(PAUSE_STATE_PATH):
            with open(PAUSE_STATE_PATH, 'r') as f:
                data = json.load(f)
                return data.get("paused", False)
    except Exception as e:
        logger.warning(f"Fehler beim Lesen des Pause-Zustands: {e}")
    return False


def set_paused(paused: bool) -> None:
    """Schreibt Pause-Zustand in Datei.
    
    Args:
        paused (bool): Neuer Pause-Zustand
    """
    try:
        with open(PAUSE_STATE_PATH, 'w') as f:
            json.dump({"paused": paused}, f)
    except Exception as e:
        logger.error(f"Fehler beim Speichern des Pause-Zustands: {e}")


def is_inside_analysis_hours() -> bool:
    """Prüft ob die aktuelle Stunde im konfigurierten Analyse-Fenster liegt.
    
    Returns:
        bool: True wenn innerhalb des Analyse-Fensters, False sonst
    """
    try:
        # Zeitzone aus Umgebung (z.B. TZ=Europe/Berlin)
        now = datetime.now()
        hour = now.hour
        # Fenster: ANALYSIS_START_HOUR <= hour < ANALYSIS_END_HOUR
        if ANALYSIS_START_HOUR <= ANALYSIS_END_HOUR:
            return ANALYSIS_START_HOUR <= hour < ANALYSIS_END_HOUR
        # z.B. 22 bis 8 (über Mitternacht)
        return hour >= ANALYSIS_START_HOUR or hour < ANALYSIS_END_HOUR
    except Exception as e:
        logger.warning(f"Fehler bei Analyse-Fenster-Prüfung: {e}")
        return True


def admin_only(handler):
    """Decorator: Nur ALLOWED_TELEGRAM_USER_ID darf den Befehl ausführen."""
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_user and update.effective_user.id != ALLOWED_TELEGRAM_USER_ID:
            await update.message.reply_text("Du bist nicht berechtigt, diesen Befehl zu nutzen.")
            return
        return await handler(update, context)
    return wrapped


class AlertManager:
    """Verwaltet Alert-Eskalation bei aufeinanderfolgenden Fehlern"""

    def __init__(self):
        self.consecutive_errors = 0
        self.max_errors_before_alert = 3
        self.last_alert_time = 0
        self.alert_cooldown = 300  # 5 Minuten Cooldown

    def on_cycle_error(self):
        """Wird bei jedem Zyklus-Fehler aufgerufen"""
        self.consecutive_errors += 1
        if self.consecutive_errors >= self.max_errors_before_alert:
            current_time = time.time()
            if current_time - self.last_alert_time > self.alert_cooldown:
                self.send_escalation_alert()
                self.last_alert_time = current_time

    def on_cycle_success(self):
        """Wird bei erfolgreichem Zyklus aufgerufen"""
        self.consecutive_errors = 0

    def send_escalation_alert(self):
        """Sendet Eskalations-Alert an Admin"""
        try:
            # Zugriff auf den Bot über globals
            if 'context' in globals() and hasattr(globals()['context'], 'bot'):
                bot = globals()['context'].bot
                if bot and ALLOWED_TELEGRAM_USER_ID:
                    import asyncio
                    asyncio.create_task(bot.send_message(
                        chat_id=ALLOWED_TELEGRAM_USER_ID,
                        text=f"⚠️ *SlopCoin Eskalation*\n\n"
                             f"SlopCoin hat {self.consecutive_errors} aufeinanderfolgende Fehler.\n"
                             f"Bitte überprüfe die Logs.",
                        parse_mode='Markdown'
                    ))
                    logger.warning("Eskalations-Alert gesendet")
        except Exception as e:
            logger.error(f"Fehler beim Senden des Eskalations-Alerts: {e}")


class HealthHandler(BaseHTTPRequestHandler):
    """HTTP Health-Check und Metrics Endpoint"""

    def do_GET(self):
        if self.path == '/health':
            try:
                last_cycle = get_last_cycle_time() if 'get_last_cycle_time' in globals() else None
                uptime = get_uptime() if 'get_uptime' in globals() else None

                response = {
                    'status': 'ok',
                    'timestamp': time.time(),
                    'last_cycle': last_cycle,
                    'uptime': uptime,
                    'container': 'SlopCoin_advisor',
                    'portfolio_size': len(portfolio) if 'portfolio' in globals() else 0
                }
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'{"status": "error", "message": "Internal error"}')
        elif self.path == '/metrics':
            # Prometheus-Format (einfach)
            try:
                metrics = []
                if 'brain' in globals() and hasattr(brain, 'cost_tracker'):
                    total_cost = brain.cost_tracker.total_cost
                    total_tokens = brain.cost_tracker.total_tokens
                    metrics.append(f"# TYPE SlopCoin_total_cost gauge")
                    metrics.append(f"SlopCoin_total_cost {total_cost}")
                    metrics.append(f"# TYPE SlopCoin_total_tokens counter")
                    metrics.append(f"SlopCoin_total_tokens {total_tokens}")

                metrics_text = "\n".join(metrics)
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(metrics_text.encode())
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b'Error')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress HTTP logs
        pass


def start_health_server(port: int = 8080) -> HTTPServer:
    """Startet den Health-Check HTTP Server in einem separaten Thread.
    
    Args:
        port (int): Port für den Health-Check Server (default: 8080)
        
    Returns:
        HTTPServer: Instanz des gestarteten HTTP Servers
    """
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health-Check Server gestartet auf Port {port}")
    return server


def get_last_cycle_time() -> Optional[float]:
    """Gibt die Zeit des letzten Analyse-Zyklus zurück.
    
    Returns:
        Optional[float]: Unix-Timestamp des letzten Zyklus oder None
    """
    try:
        history = risk_analyzer._load_history() if 'risk_analyzer' in globals() else {}
        return history.get('last_cycle_timestamp')
    except:
        return None


def get_uptime() -> Optional[float]:
    """Gibt die Uptime in Sekunden zurück.
    
    Returns:
        Optional[float]: Uptime in Sekunden oder None
    """
    try:
        if 'start_time' in globals():
            return time.time() - start_time
        return None
    except:
        return None


@admin_only
async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Befehl: /help"""
    text = (
        "*SlopCoin v1.0 – Befehle*\n\n"
        "/status – Aktuellen Portfolio-Status abrufen\n"
        "/dashboard – Visuelle Portfolio-Allokation\n"
        "/heatmap – Korrelationsmatrix\n"
        "/what\\_if <COIN> <+/-%> – Szenario-Analyse\n"
        "/next <EUR> – Investment-Empfehlung für neuen Betrag\n"
        "/pause – Automatische Analyse pausieren\n"
        "/resume – Automatische Analyse wieder starten\n"
        "/help – Diese Hilfe anzeigen\n\n"
        "*Automatische Funktionen:*\n"
        "• Tägliche KI-Analyse (nur bei Signal, 08:00–22:00)\n"
        "• Preis-Alerts alle 30 Min (kostenlos, kein KI-Call)\n"
        "• Wöchentliche Management-Summary jeden Sonntag 10:00\n\n"
        "Hinweis: Nachrichten werden nur zwischen 08:00 und 22:00 gesendet (konfigurierbar)."
    )
    await update.message.reply_text(text, parse_mode='Markdown')


@admin_only
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Befehl: /status – Portfolio-Status abfragen (ohne KI, nur Daten)."""
    await update.message.reply_text("Lade Portfolio-Status…")
    try:
        portfolio = market.get_portfolio()
        if not portfolio:
            await update.message.reply_text("Portfolio ist leer oder konnte nicht geladen werden.")
            return

        _, prices = market.get_portfolio_with_prices()

        # Coins ohne gültigen Preis (None oder 0) für Logging erfassen
        invalid_price_coins = {c: prices.get(c) for c in portfolio if not prices.get(c)}
        if invalid_price_coins:
            logger.warning(f"/status: Coins ohne gültigen Preis: {invalid_price_coins}")

        # Nur Coins mit gültigem Preis für Wertberechnungen verwenden
        valid_portfolio_items = [
            (c, a) for c, a in portfolio.items() if prices.get(c) not in (None, 0)
        ]

        total_eur = sum(amount * prices.get(coin, 0) for coin, amount in valid_portfolio_items)

        lines = [f"*Portfolio-Status* (Stand: {datetime.now().strftime('%d.%m.%Y %H:%M')})\n"]
        lines.append(f"*Gesamtwert:* {total_eur:.2f} EUR\n")

        if tracker.has_baseline():
            baseline = tracker.load_baseline()
            perf = tracker.calculate_performance(portfolio, prices, baseline)
            if perf:
                lines.append(f"*ROI (vs. Baseline):* {perf['total_roi_percent']:+.2f}%\n")
                if perf.get('best_performer') and perf['best_performer'].get('coin'):
                    lines.append(f"*Bester Performer:* {perf['best_performer']['coin']} ({perf['best_performer']['roi_percent']:+.2f}%)\n")
                if perf.get('worst_performer') and perf['worst_performer'].get('coin'):
                    lines.append(f"*Schlechtester Performer:* {perf['worst_performer']['coin']} ({perf['worst_performer']['roi_percent']:+.2f}%)\n")

        lines.append("\n*Positionen:*")
        # Nur valide Positionen sortieren; None-Preise werden ausgeschlossen
        for coin, amount in sorted(
            valid_portfolio_items,
            key=lambda x: -(x[1] * (prices.get(x[0]) or 0)),
        ):
            p = prices.get(coin)
            if p is None:
                continue
            val = amount * p
            pct = (val / total_eur * 100) if total_eur else 0
            lines.append(f"• {coin}: {amount:.4f} ≈ {val:.2f} EUR ({pct:.1f}%)")

        # Hinweis für Coins ohne Preis ans Ende der Nachricht hängen
        if invalid_price_coins:
            missing_list = ", ".join(sorted(invalid_price_coins.keys()))
            lines.append(
                f"\n_Hinweis: Für folgende Coins sind aktuell keine Preise verfügbar und sie wurden in der Aufstellung ignoriert: {missing_list}_"
            )

        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
    except Exception as e:
        logger.exception("Fehler bei /status")
        await update.message.reply_text(f"Fehler beim Laden des Status: {str(e)}")


@admin_only
async def cmd_dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Befehl: /dashboard – Visuelle Portfolio-Übersicht"""
    try:
        portfolio = market.get_portfolio()
        if not portfolio:
            await update.message.reply_text("Portfolio ist leer.")
            return

        _, prices = market.get_portfolio_with_prices()
        total_eur = sum(portfolio.get(c, 0) * prices.get(c, 0) for c in portfolio if prices.get(c))

        # Portfolio-Gewichtungen berechnen
        weights = {}
        for coin in portfolio:
            if coin in prices and prices[coin]:
                weights[coin] = (portfolio[coin] * prices[coin]) / total_eur * 100

        # ASCII Chart generieren
        lines = ["*Portfolio Allokation:*\n"]
        width = 40  # Balkenbreite

        for coin, pct in sorted(weights.items(), key=lambda x: -x[1]):
            bar_length = int(width * pct / 100)
            bar = '█' * bar_length
            lines.append(f"{coin:6} {bar} {pct:5.1f}%")

        # Performance-Info hinzufügen
        if tracker.has_baseline():
            baseline = tracker.load_baseline()
            perf = tracker.calculate_performance(portfolio, prices, baseline)
            if perf:
                lines.append(f"\n*ROI:* {perf['total_roi_percent']:+.2f}%")
                if perf.get('best_performer'):
                    lines.append(f"*Bester:* {perf['best_performer']['coin']} ({perf['best_performer']['roi_percent']:+.2f}%)")

        lines.append(f"\n*Gesamtwert:* {total_eur:.2f} EUR")
        lines.append(f"*Coins:* {len(portfolio)}")

        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
    except Exception as e:
        logger.exception("Fehler bei /dashboard")
        await update.message.reply_text(f"Fehler: {str(e)}")


@admin_only
async def cmd_heatmap(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Befehl: /heatmap – Korrelationsmatrix als Text-Heatmap"""
    try:
        # Preis-Historie laden
        history = risk_analyzer._load_history()
        price_history_dict = history.get('price_history', {})

        if len(price_history_dict) < 2:
            await update.message.reply_text("Nicht genug Daten für Korrelationsmatrix (mind. 2 Coins benötigt).")
            return

        # Korrelationsmatrix berechnen
        portfolio_coins = list(portfolio.keys()) if 'portfolio' in locals() else list(price_history_dict.keys())
        corr_matrix = risk_analyzer.calculate_correlation_matrix(portfolio_coins, price_history_dict)

        if not corr_matrix:
            await update.message.reply_text("Korrelationsmatrix konnte nicht berechnet werden.")
            return

        # Text-basierte Heatmap erstellen
        coins = list(corr_matrix.keys())
        lines = ["*Korrelationsmatrix:*\n"]
        lines.append("     " + "   ".join(f"{c:>5}" for c in coins))

        for coin1 in coins:
            row = [f"{coin1:>5}"]
            for coin2 in coins:
                corr = corr_matrix.get(coin1, {}).get(coin2, 0)
                # Farb-Codierung (einfach mit Zeichen)
                if corr >= 0.8:
                    bar = "██"  # Sehr hoch
                elif corr >= 0.6:
                    bar = "▓▓"  # Hoch
                elif corr >= 0.4:
                    bar = "▒▒"  # Mittel
                elif corr >= 0.2:
                    bar = "░░"  # Niedrig
                else:
                    bar = "  "  # Sehr niedrig/negativ
                row.append(f"{bar}{corr:>4.2f}")
            lines.append(" ".join(row))

        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
    except Exception as e:
        logger.exception("Fehler bei /heatmap")
        await update.message.reply_text(f"Fehler: {str(e)}")


@admin_only
async def cmd_what_if(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Befehl: /what_if <COIN> <AMOUNT> <TARGET_PRICE> – Szenario-Analyse"""
    try:
        # Input-Validierung mit Pydantic
        valid, request, error_msg = validate_what_if_args(context.args)
        if not valid:
            await update.message.reply_text(error_msg)
            return

        coin = request.coin
        change_percent = request.change_percent
        change_pct = change_percent / 100  # In Dezimal umwandeln

        portfolio = market.get_portfolio()
        if coin not in portfolio:
            await update.message.reply_text(f"{coin} nicht im Portfolio.")
            return

        _, prices = market.get_portfolio_with_prices()
        total_old = sum(portfolio.get(c, 0) * prices.get(c, 0) for c in portfolio if prices.get(c))

        # Szenario berechnen
        new_portfolio = portfolio.copy()
        if coin in new_portfolio:
            new_amount = new_portfolio[coin] * (1 + change_pct)
            if new_amount < 0.001:  # Kraken Mindestmenge
                await update.message.reply_text(f"⚠️ {coin} Menge würde unter 0.001 fallen – nicht erlaubt.")
                return
            new_portfolio[coin] = new_amount

        # Neuen Gesamtwert berechnen (Preise bleiben gleich)
        total_new = sum(new_portfolio.get(c, 0) * prices.get(c, 0) for c in new_portfolio if prices.get(c))
        change_eur = total_new - total_old
        change_pct_total = (change_eur / total_old * 100) if total_old > 0 else 0

        lines = [
            "*Szenario-Analyse*\n",
            f"Original: {portfolio[coin]:.4f} {coin}",
            f"Neu: {new_portfolio[coin]:.4f} {coin} ({change_percent:+.1f}%)",
            f"\nPortfolio-Wert:",
            f"  Vorher: {total_old:.2f} EUR",
            f"  Nachher: {total_new:.2f} EUR",
            f"  Änderung: {change_eur:+.2f} EUR ({change_pct_total:+.2f}%)"
        ]

        await update.message.reply_text("\n".join(lines), parse_mode='Markdown')
    except Exception as e:
        logger.exception("Fehler bei /what_if")
        await update.message.reply_text(f"Fehler: {str(e)}")


@admin_only
async def cmd_next(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Befehl: /next <EUR> – KI-gestützte Investment-Empfehlung für einen neuen Betrag.

    Analysiert das bestehende Portfolio, aktuelle Marktdaten und technische
    Indikatoren, um zu empfehlen, wo ein neuer EUR-Betrag am besten investiert
    werden soll. Die Anzahl und Größe der Splits entscheidet die KI selbst.

    Beispiele:
        /next 50   → Empfehlung für 50 EUR (ggf. alles in einen Coin)
        /next 500  → Empfehlung für 500 EUR (ggf. 3-5 Splits)
        /next 5000 → Empfehlung für 5.000 EUR (ggf. breite Diversifikation)
    """
    # 1. Input-Validierung
    valid, request, error_msg = validate_next_invest_args(context.args)
    if not valid:
        await update.message.reply_text(error_msg)
        return

    invest_amount = request.amount

    await update.message.reply_text(
        f"🔍 Analysiere Investitionsmöglichkeiten für *{invest_amount:,.2f} EUR*…\n"
        f"_(Portfolio, Marktdaten & Web-Search werden ausgewertet)_",
        parse_mode='Markdown'
    )

    try:
        # 2. Portfolio & Marktdaten laden
        portfolio = market.get_portfolio()
        if not portfolio:
            await update.message.reply_text("⚠️ Portfolio ist leer oder konnte nicht geladen werden.")
            return

        portfolio_with_prices, prices = market.get_portfolio_with_prices()
        portfolio_indicators = market.get_portfolio_indicators(portfolio)
        exclude_coins = list(portfolio.keys())
        market_overview = market.get_market_overview(top_n=20, exclude_coins=exclude_coins)

        # Performance-Daten laden (optional, für Kontext)
        performance_data = None
        if tracker.has_baseline():
            baseline = tracker.load_baseline()
            performance_data = tracker.calculate_performance(portfolio_with_prices, prices, baseline)

        # 3. KI-Analyse (Analyst → Guardian)
        result = brain.analyze_next_investment(
            invest_amount=invest_amount,
            portfolio_data=portfolio_with_prices,
            portfolio_indicators=portfolio_indicators,
            market_overview=market_overview,
            performance_data=performance_data,
            cycle_num=int(time.time() // 3600),  # Stündliche Zyklus-Nummer
        )

        if result is None:
            await update.message.reply_text(
                "❌ Die KI-Analyse ist fehlgeschlagen. Bitte versuche es später erneut."
            )
            return

        if result.get('approved'):
            try:
                await update.message.reply_text(
                    result['message'],
                    parse_mode='Markdown'
                )
                logger.info(
                    f"/next {invest_amount} EUR – Empfehlung gesendet "
                    f"({result.get('total_splits', '?')} Splits, "
                    f"Strategie: {result.get('strategy', 'UNKNOWN')})"
                )
            except Exception as e:
                logger.error(f"Fehler beim Senden der /next Antwort: {e}")
                # Fallback: Nachricht ohne Markdown senden
                await update.message.reply_text(result['message'])
        else:
            await update.message.reply_text(
                f"⚠️ *Guardian hat die Empfehlung abgelehnt*\n\n{result.get('message', 'Keine Details verfügbar.')}",
                parse_mode='Markdown'
            )

    except Exception as e:
        logger.exception("Fehler bei /next")
        await update.message.reply_text(f"❌ Fehler bei der Analyse: {str(e)}")


@admin_only
async def cmd_set_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Befehl: /set_interval <hours> – Ändert Analyse-Intervall (erfordert Container-Neustart)"""
    try:
        # Input-Validierung mit Pydantic
        valid, request, error_msg = validate_set_interval_args(context.args)
        if not valid:
            await update.message.reply_text(error_msg)
            return

        hours = request.hours

        # Nur Bestätigung, da Environment Variable erst nach Neustart wirksam wird
        await update.message.reply_text(
            f"Intervall auf {hours}h gesetzt.\n\n"
            f"⚠️ *Hinweis:* Container muss neu gestartet werden, damit die Änderung wirksam wird.\n"
            f"Setze Environment Variable: SCHEDULE_INTERVAL_HOURS={hours}",
            parse_mode='Markdown'
        )
    except (IndexError, ValueError):
        await update.message.reply_text("Usage: /set_interval <hours>")


@admin_only
async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Befehl: /pause"""
    set_paused(True)
    await update.message.reply_text(
        "Automatische Analyse ist *pausiert*. Du bekommst keine geplanten Abfragen mehr. "
        "Mit /resume wieder starten.",
        parse_mode='Markdown'
    )


@admin_only
async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Befehl: /resume"""
    set_paused(False)
    await update.message.reply_text(
        "Automatische Analyse läuft wieder. Du bekommst wie gewohnt Empfehlungen im Analyse-Fenster.",
        parse_mode='Markdown'
    )


async def run_price_alert_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Preis-Alert-Check alle 30 Min — ohne LLM, nur Schwellenwert-Mathematik.

    Sendet sofort einen Alert wenn ein Portfolio-Coin die konfigurierten
    Schwellenwerte (PRICE_ALERT_THRESHOLD_UP / DOWN) überschreitet.
    Kein LLM-Call, keine Kosten.

    Args:
        context (ContextTypes.DEFAULT_TYPE): Telegram-Kontext
    """
    if is_paused():
        return

    try:
        portfolio = market.get_portfolio()
        if not portfolio:
            return

        if not tracker.has_baseline():
            return  # Noch keine Baseline → kein Vergleich möglich

        _, prices = market.get_portfolio_with_prices()
        baseline = tracker.load_baseline()
        # calculate_performance erwartet {coin: amount} — direkt portfolio übergeben
        performance_data = tracker.calculate_performance(portfolio, prices, baseline)

        if not performance_data or not performance_data.get('coin_performance'):
            return

        alerts = []
        for coin, perf in performance_data['coin_performance'].items():
            roi = perf.get('roi_percent', 0) / 100  # In Dezimal
            if roi >= PRICE_ALERT_THRESHOLD_UP:
                alerts.append(
                    f"🚀 *TAKE-PROFIT SIGNAL*: {coin} ist um "
                    f"*{roi * 100:+.1f}%* gestiegen (Schwellenwert: +{PRICE_ALERT_THRESHOLD_UP * 100:.0f}%)\n"
                    f"   Aktueller Preis: {prices.get(coin, 0):.4f} EUR"
                )
            elif roi <= -PRICE_ALERT_THRESHOLD_DOWN:
                alerts.append(
                    f"🔴 *STOP-LOSS SIGNAL*: {coin} ist um "
                    f"*{roi * 100:+.1f}%* gefallen (Schwellenwert: -{PRICE_ALERT_THRESHOLD_DOWN * 100:.0f}%)\n"
                    f"   Aktueller Preis: {prices.get(coin, 0):.4f} EUR"
                )

        if alerts:
            msg = (
                "⚡ *Preis-Alert* (automatisch, kein KI-Call)\n\n"
                + "\n\n".join(alerts)
                + "\n\n_Für vollständige KI-Analyse: /analyse_"
            )
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=msg,
                    parse_mode='Markdown'
                )
                logger.info(f"Preis-Alert gesendet: {len(alerts)} Signal(e)")
            except Exception as e:
                logger.error(f"Fehler beim Senden des Preis-Alerts: {e}")

    except Exception as e:
        logger.error(f"Preis-Alert-Check Fehler: {e}", exc_info=True)


async def run_weekly_summary(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Wöchentliche Management-Summary — jeden Sonntag um WEEKLY_SUMMARY_HOUR Uhr.

    Wird stündlich geprüft. Sendet IMMER eine Nachricht (auch bei HOLD).
    Kein Guardian-Call — Kosteneinsparung ~50% gegenüber täglicher Analyse.

    Args:
        context (ContextTypes.DEFAULT_TYPE): Telegram-Kontext
    """
    if is_paused():
        return

    now = datetime.now()

    # Nur am konfigurierten Wochentag (Standard: Sonntag = 6) ausführen
    if now.weekday() != WEEKLY_SUMMARY_DAY:
        return

    # Nur in der konfigurierten Stunde ausführen (Fenster: HH:00–HH:59)
    if now.hour != WEEKLY_SUMMARY_HOUR:
        return

    logger.info(f"Starte wöchentliche Management-Summary (KW {now.isocalendar()[1]})…")

    try:
        portfolio = market.get_portfolio()
        if not portfolio:
            logger.warning("Weekly Summary: Portfolio ist leer")
            return

        portfolio_with_prices, prices = market.get_portfolio_with_prices()

        # Baseline sicherstellen
        if not tracker.has_baseline():
            logger.info("Weekly Summary: Keine Baseline – erstelle Baseline…")
            tracker.save_baseline(portfolio_with_prices, prices)
            total = sum(portfolio_with_prices.get(c, 0) * prices.get(c, 0) for c in portfolio_with_prices if prices.get(c))
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=(
                        f"*SlopCoin – Baseline erstellt*\n\n"
                        f"Portfolio-Wert: {total:.2f} EUR\n"
                        f"Wöchentliche Summary startet nächsten Sonntag."
                    ),
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Fehler beim Senden der Baseline-Nachricht: {e}")
            return

        baseline = tracker.load_baseline()
        performance_data = tracker.calculate_performance(portfolio_with_prices, prices, baseline)
        portfolio_indicators = market.get_portfolio_indicators(portfolio)
        risk_metrics = risk_analyzer.analyze_risks(
            portfolio_with_prices, prices, portfolio_indicators, performance_data
        )
        exclude_coins = list(portfolio.keys())
        market_overview = market.get_market_overview(top_n=20, exclude_coins=exclude_coins)

        # KI-Analyse — kein Guardian (Kosteneinsparung)
        result = brain.analyze_weekly_summary(
            portfolio_data=portfolio_with_prices,
            portfolio_indicators=portfolio_indicators,
            market_overview=market_overview,
            performance_data=performance_data,
            risk_metrics=risk_metrics,
            cycle_num=int(now.isocalendar()[1]),  # Kalenderwoche als Zyklus-Nummer
        )

        if result is None:
            logger.error("Weekly Summary: KI-Analyse fehlgeschlagen")
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text="⚠️ *Wöchentliche Summary fehlgeschlagen*\n\nDie KI-Analyse konnte nicht durchgeführt werden. Bitte Logs prüfen.",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Fehler beim Senden der Fehler-Nachricht: {e}")
            return

        # Immer senden — auch bei HOLD
        try:
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=result['message'],
                parse_mode='Markdown'
            )
            logger.info(
                f"Weekly Summary gesendet (KW {now.isocalendar()[1]}, "
                f"Empfehlung: {result.get('recommendation', 'HOLD')}, "
                f"Sentiment: {result.get('sentiment', 'neutral')})"
            )
        except Exception as e:
            logger.error(f"Fehler beim Senden der Weekly Summary: {e}")
            # Fallback ohne Markdown
            try:
                await context.bot.send_message(chat_id=ADMIN_ID, text=result['message'])
            except Exception as e2:
                logger.error(f"Fallback-Senden fehlgeschlagen: {e2}")

    except Exception as e:
        logger.error(f"Weekly Summary Fehler: {e}", exc_info=True)
        try:
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=f"*Fehler in der wöchentlichen Summary*\n\n{str(e)}",
                parse_mode='Markdown'
            )
        except Exception:
            pass


async def run_cycle(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Führt den geplanten Haupt-Analyse-Zyklus aus.
    
    Args:
        context (ContextTypes.DEFAULT_TYPE): Telegram-Kontext
        
    Returns:
        None
    """
    global alert_manager

    if is_paused():
        logger.info("Zyklus übersprungen: Bot ist pausiert")
        alert_manager.on_cycle_success()
        return
    if not is_inside_analysis_hours():
        logger.info("Zyklus übersprungen: außerhalb des Analyse-Fensters")
        alert_manager.on_cycle_success()
        return

    logger.info("Starte Analyse-Zyklus…")
    try:
        portfolio = market.get_portfolio()
        if not portfolio:
            logger.warning("Portfolio ist leer")
            alert_manager.on_cycle_success()
            return

        portfolio_with_prices, prices = market.get_portfolio_with_prices()

        if not tracker.has_baseline():
            logger.info("Keine Baseline – erstelle Baseline…")
            tracker.save_baseline(portfolio_with_prices, prices)
            total = sum(portfolio_with_prices.get(c, 0) * prices.get(c, 0) for c in portfolio_with_prices if prices.get(c))
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=f"*SlopCoin v0.1.0 gestartet*\n\nBaseline erstellt: Portfolio-Wert {total:.2f} EUR\n\nPerformance-Tracking startet beim nächsten Lauf.",
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Fehler beim Senden der Start-Nachricht: {e}")
            alert_manager.on_cycle_success()
            return

        baseline = tracker.load_baseline()
        performance_data = tracker.calculate_performance(portfolio_with_prices, prices, baseline)
        portfolio_indicators = market.get_portfolio_indicators(portfolio)
        risk_metrics = risk_analyzer.analyze_risks(
            portfolio_with_prices, prices, portfolio_indicators, performance_data
        )
        exclude_coins = list(portfolio.keys())
        market_overview = market.get_market_overview(top_n=20, exclude_coins=exclude_coins)

        result = brain.analyze_market(
            portfolio_data=portfolio_with_prices,
            portfolio_indicators=portfolio_indicators,
            market_overview=market_overview,
            performance_data=performance_data,
            risk_metrics=risk_metrics,
            cycle_num=int(time.time() // 86400),  # Zyklus-Nummer basierend auf 24h-Intervall
            news_context=None  # Kein vorgeladener News-Kontext → Modell nutzt Web-Search
        )

        if result and result.get('approved'):
            try:
                await context.bot.send_message(
                    chat_id=ADMIN_ID,
                    text=result['message'],
                    parse_mode='Markdown'
                )
                logger.info("Alert gesendet")
            except Exception as e:
                logger.error(f"Fehler beim Senden: {e}")
        else:
            if result and not result.get('approved'):
                logger.warning("Guardian hat Empfehlung abgelehnt")
            else:
                logger.info("Keine Handlung erforderlich")

        alert_manager.on_cycle_success()

    except Exception as e:
        logger.error(f"Cycle Error: {e}", exc_info=True)
        alert_manager.on_cycle_error()
        try:
            await context.bot.send_message(
                chat_id=ADMIN_ID,
                text=f"*Fehler im Analyse-Zyklus*\n\n{str(e)}",
                parse_mode='Markdown'
            )
        except Exception:
            pass


def main() -> None:
    """Haupt-Einstiegspunkt von SlopCoin.
    
    Initialisiert alle Komponenten, validiert Konfiguration und startet den Bot.
    """
    global market, tracker, risk_analyzer, brain, ADMIN_ID, alert_manager, start_time, health_server

    # Configuration Validation beim Start
    logger.info("🔍 Validiere Konfiguration…")
    valid, message = ConfigValidator.validate_all_configurations()
    if not valid:
        logger.critical(f"Konfigurations-Validierung fehlgeschlagen: {message}")
        raise SystemExit(1)
    logger.info(f"✅ Konfiguration validiert: {message}")

    # Signal Handler für Graceful Shutdown setup
    setup_signal_handlers()
    
    # Graceful Shutdown Cleanup-Funktionen registrieren
    def cleanup_on_shutdown():
        logger.info("Führe Cleanup während Shutdown durch…")
        # Hier könnten weitere Cleanup-Aktionen hinzugefügt werden
        # z.B. Cache leeren, Verbindungen schließen, etc.
    
    register_cleanup_function(cleanup_on_shutdown)

    try:
        with open(TELEGRAM_TOKEN_PATH, 'r') as f:
            token = f.read().strip()
    except FileNotFoundError:
        logger.critical(f"Telegram Token nicht gefunden: {TELEGRAM_TOKEN_PATH}")
        raise

    # Admin-ID ist hardcodiert
    ADMIN_ID = ALLOWED_TELEGRAM_USER_ID

    logger.info("Initialisiere Komponenten…")
    market = MarketData()
    tracker = PortfolioTracker()
    risk_analyzer = RiskAnalyzer()
    brain = LLMEngine()
    alert_manager = AlertManager()
    logger.info("Alle Komponenten initialisiert")

    # Health-Check Server starten
    start_health_server(port=8080)
    start_time = time.time()

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("dashboard", cmd_dashboard))
    app.add_handler(CommandHandler("heatmap", cmd_heatmap))
    app.add_handler(CommandHandler("what_if", cmd_what_if))
    app.add_handler(CommandHandler("next", cmd_next))
    app.add_handler(CommandHandler("set_interval", cmd_set_interval))
    app.add_handler(CommandHandler("pause", cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))

    # Geplante Jobs registrieren
    job_queue = app.job_queue
    if job_queue:
        # 1x/Tag Deep-Analysis mit KI (Analyst + Guardian + Web-Search) — nur bei Signal
        job_queue.run_repeating(run_cycle, interval=SCHEDULE_INTERVAL_SECONDS, first=10)
        logger.info(
            f"Geplante KI-Analyse: alle {SCHEDULE_INTERVAL_SECONDS // 3600}h, "
            f"nur zwischen {ANALYSIS_START_HOUR:02d}:00 und {ANALYSIS_END_HOUR:02d}:00 (nur bei Signal)"
        )
        # Preis-Alerts alle 30 Min (kein LLM, kostenlos)
        job_queue.run_repeating(
            run_price_alert_check,
            interval=PRICE_CHECK_INTERVAL_SECONDS,
            first=60  # Erster Check nach 1 Minute
        )
        logger.info(
            f"Preis-Alert-Check: alle {PRICE_CHECK_INTERVAL_SECONDS // 60} Minuten "
            f"(Schwellenwerte: +{PRICE_ALERT_THRESHOLD_UP * 100:.0f}% / -{PRICE_ALERT_THRESHOLD_DOWN * 100:.0f}%)"
        )
        # Wöchentliche Management-Summary (stündlich prüfen ob Sonntag 10:00)
        # Kein Guardian — Kosteneinsparung ~50%
        WEEKLY_CHECK_INTERVAL = 3600  # Stündlich prüfen
        day_names = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
        job_queue.run_repeating(
            run_weekly_summary,
            interval=WEEKLY_CHECK_INTERVAL,
            first=90  # Erster Check nach 90 Sekunden
        )
        logger.info(
            f"Wöchentliche Summary: jeden {day_names[WEEKLY_SUMMARY_DAY]} "
            f"um {WEEKLY_SUMMARY_HOUR:02d}:00 Uhr (kein Guardian)"
        )
    else:
        logger.warning("JobQueue nicht verfügbar – keine automatische Analyse")

    logger.info("SlopCoin v0.1.0 startet – Telegram-Befehle: /status, /dashboard, /heatmap, /what_if, /next, /set_interval, /pause, /resume, /help")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
