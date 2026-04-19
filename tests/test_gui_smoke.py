import pytest

from library_escape.gui.app import LauncherApp


@pytest.fixture(scope="module")
def app():
    app = LauncherApp()
    app.update_idletasks()
    app.update()
    yield app
    try:
        app.destroy()
    except Exception:
        pass


def test_launcher_app_smoke(app):
    assert app.play_frame.deterministic_policy_var.get() is True
    assert app.results_frame.deterministic_playback_var.get() is True


def test_train_logs_panel_has_readable_space(app):
    app.geometry("1366x820")
    app.notebook.select(app.train_frame)
    app.update_idletasks()
    app.update()
    app.update_idletasks()
    assert app.train_frame.log_text.winfo_width() >= 420
    assert app.train_frame.log_text.winfo_height() >= 180


def test_train_sidebar_has_scrollbar_for_overflowing_controls(app):
    app.geometry("1280x720")
    app.notebook.select(app.train_frame)
    app.update_idletasks()
    app.update()
    app.update_idletasks()
    assert app.train_frame.setup_scroll.v_scrollbar.winfo_ismapped()
    assert app.train_frame.setup_scroll.content.winfo_reqheight() > app.train_frame.setup_scroll.canvas.winfo_height()
