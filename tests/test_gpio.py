import pytest
import sys
import os
import importlib.util
from unittest.mock import patch, MagicMock

# Dynamically import the gpio module to avoid importing app.core.__init__ which imports ultralytics etc
spec = importlib.util.spec_from_file_location("gpio", os.path.join(os.path.dirname(os.path.dirname(__file__)), "app", "core", "gpio.py"))
gpio = importlib.util.module_from_spec(spec)
sys.modules["app.core.gpio"] = gpio
spec.loader.exec_module(gpio)


@pytest.fixture(autouse=True)
def reset_gpio_state():
    """Reset the GPIO module state before each test."""
    gpio._led = None
    gpio._available = False
    yield
    gpio._led = None
    gpio._available = False

def test_init_success():
    """Test successful initialization when gpiozero is available."""
    # Create a mock for gpiozero module and its LED class
    mock_gpiozero = MagicMock()
    mock_led_class = MagicMock()
    mock_gpiozero.LED = mock_led_class

    # Mock sys.modules to return our mock gpiozero when imported
    with patch.dict('sys.modules', {'gpiozero': mock_gpiozero}):
        # Mock print to verify output
        with patch('builtins.print') as mock_print:
            gpio.init(pin=17)

            # Verify LED was instantiated with the correct pin
            mock_led_class.assert_called_once_with(17)

            # Verify state was updated
            assert gpio._available is True
            assert gpio._led is not None

            # Verify correct output was printed
            mock_print.assert_called_once_with("GPIO Pin 17 hazir")

def test_init_import_error():
    """Test initialization behavior when gpiozero is not available (ImportError)."""
    # Force ImportError when importing gpiozero
    with patch.dict('sys.modules', {'gpiozero': None}):
        with patch('builtins.print') as mock_print:
            gpio.init(pin=17)

            # Verify state reflects failure
            assert gpio._available is False
            assert gpio._led is None

            # Verify correct output was printed
            mock_print.assert_called_once_with("GPIO yok, LED devre disi")

def test_on():
    """Test the 'on' function correctly calls the underlying LED's 'on' method."""
    mock_led = MagicMock()
    gpio._led = mock_led
    gpio._available = True

    gpio.on()

    mock_led.on.assert_called_once()

def test_on_not_available():
    """Test the 'on' function does nothing when GPIO is not available."""
    mock_led = MagicMock()
    gpio._led = mock_led
    gpio._available = False

    gpio.on()

    mock_led.on.assert_not_called()

def test_on_no_led():
    """Test the 'on' function does nothing when LED is None."""
    gpio._led = None
    gpio._available = True

    gpio.on()

def test_off():
    """Test the 'off' function correctly calls the underlying LED's 'off' method."""
    mock_led = MagicMock()
    gpio._led = mock_led
    gpio._available = True

    gpio.off()

    mock_led.off.assert_called_once()

def test_off_not_available():
    """Test the 'off' function does nothing when GPIO is not available."""
    mock_led = MagicMock()
    gpio._led = mock_led
    gpio._available = False

    gpio.off()

    mock_led.off.assert_not_called()

def test_off_no_led():
    """Test the 'off' function does nothing when LED is None."""
    gpio._led = None
    gpio._available = True

    gpio.off()

def test_is_available():
    """Test the 'is_available' function correctly returns the current availability state."""
    gpio._available = True
    assert gpio.is_available() is True

    gpio._available = False
    assert gpio.is_available() is False
