from unittest.mock import MagicMock


class Menu:
    def __init__(self, title=""):
        self.items = []
        self.name = title

    def numberOfItems(self):
        return len(self.items)

    def itemAtIndex_(self, index):
        return self.items[index]

    def addItem_(self, item):
        self.items.append(item)


class Item:
    def __init__(self, title="", action=None, key=""):
        self.name, self.action, self.key = title, action, key
        self.menu = None
        self.target = "unset"

    def title(self):
        return self.name

    def setTitle_(self, title):
        self.name = title

    def setSubmenu_(self, menu):
        self.menu = menu

    def submenu(self):
        return self.menu

    def setTarget_(self, target):
        self.target = target


def test_command_w_uses_native_close_action_and_is_installed_once(main_module, monkeypatch):
    import AppKit

    root = Menu()
    app = MagicMock()
    app.mainMenu.return_value = root
    monkeypatch.setattr(AppKit, "NSApp", lambda: app)
    monkeypatch.setattr(AppKit.NSMenu, "new", lambda: Menu())
    monkeypatch.setattr(AppKit.NSMenu, "alloc", lambda: MagicMock(initWithTitle_=lambda title: Menu(title)))
    monkeypatch.setattr(AppKit.NSMenuItem, "new", lambda: Item())
    monkeypatch.setattr(AppKit.NSMenuItem, "alloc", lambda: MagicMock(initWithTitle_action_keyEquivalent_=lambda *args: Item(*args)))

    main_module._ensure_edit_menu()
    main_module._ensure_edit_menu()
    file_items = [item for item in root.items if item.title() == "File"]
    assert len(file_items) == 1
    close, = file_items[0].submenu().items
    assert (close.name, close.action, close.key, close.target) == ("Close", "performClose:", "w", None)
    assert len([item for item in root.items if item.title() == "Edit"]) == 1


def test_close_menu_is_added_when_edit_menu_already_exists(main_module, monkeypatch):
    import AppKit

    root = Menu()
    existing = Item("Edit")
    root.addItem_(existing)
    app = MagicMock()
    app.mainMenu.return_value = root
    monkeypatch.setattr(AppKit, "NSApp", lambda: app)
    monkeypatch.setattr(AppKit.NSMenu, "alloc", lambda: MagicMock(initWithTitle_=lambda title: Menu(title)))
    monkeypatch.setattr(AppKit.NSMenuItem, "new", lambda: Item())
    monkeypatch.setattr(AppKit.NSMenuItem, "alloc", lambda: MagicMock(initWithTitle_action_keyEquivalent_=lambda *args: Item(*args)))
    main_module._ensure_edit_menu()
    assert root.items[0] is existing
    assert [item.title() for item in root.items] == ["Edit", "File"]
