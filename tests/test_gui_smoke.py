from library_escape.gui.app import LauncherApp


def test_launcher_app_smoke():
    app = LauncherApp()
    try:
        app.update_idletasks()
        app.update()
        assert app.play_frame.deterministic_policy_var.get() is True
        assert app.results_frame.deterministic_playback_var.get() is True
    finally:
        app.destroy()
