# GPIO LED kontrolu
_led = None
_available = False

def init(pin=17):
    global _led, _available
    try:
        from gpiozero import LED
        _led = LED(pin)
        _available = True
        print(f"GPIO Pin {pin} hazir")
    except ImportError:
        _available = False
        print("GPIO yok, LED devre disi")

def on():
    if _available and _led:
        _led.on()

def off():
    if _available and _led:
        _led.off()

def is_available():
    return _available
