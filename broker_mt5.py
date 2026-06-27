"""MT4/MT5 broker integration stub. Enable by setting broker.enabled=true in config.yaml."""
import os
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "config.yaml")) as f:
    CFG = yaml.safe_load(f)

MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    pass


def is_enabled():
    return CFG.get("broker", {}).get("enabled", False) and MT5_AVAILABLE


def connect():
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 package not installed")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
    return True


def place_order(direction, entry, sl, tp1, tp2, lot, symbol="XAUUSD"):
    """Place order on MT5. Returns result dict or None."""
    if not is_enabled():
        return None
    connect()
    order_type = mt5.ORDER_TYPE_BUY if direction.startswith("BUY") else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(symbol)
    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol, "volume": lot, "type": order_type,
        "price": price, "sl": sl, "tp": tp1,
        "deviation": 10, "magic": 20260627, "comment": "XAUUSD_BOT",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    mt5.shutdown()
    return {"ticket": result.order, "price": result.price, "retcode": result.retcode}


def close_position(ticket, symbol="XAUUSD"):
    """Close position by ticket."""
    if not is_enabled():
        return None
    connect()
    pos = mt5.positions_get(ticket=ticket)
    if not pos:
        mt5.shutdown()
        return None
    pos = pos[0]
    order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol,
        "volume": pos.volume, "type": order_type, "position": ticket,
        "price": price, "deviation": 10, "magic": 20260627, "comment": "CLOSE",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    mt5.shutdown()
    return {"retcode": result.retcode}


def get_account_info():
    if not is_enabled():
        return None
    connect()
    info = mt5.account_info()
    mt5.shutdown()
    return {"balance": info.balance, "equity": info.equity, "margin": info.margin}
