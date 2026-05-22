from pynput.keyboard import Controller, Key

_kb = Controller()


def press_space_down() -> None:
    """Hold the spacebar down. Must be paired with press_space_up()."""
    _kb.press(Key.space)


def press_space_up() -> None:
    """Release the spacebar. Safe to call even if not currently pressed."""
    _kb.release(Key.space)
