from library_escape.gui.app import LauncherApp


def test_launcher_app_smoke():
    app = LauncherApp()
    try:
        app.update_idletasks()
        app.update()
    finally:
        app.destroy()
