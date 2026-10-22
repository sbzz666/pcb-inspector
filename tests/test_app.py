from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_pages_render_without_exception():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
    assert not app.exception
    for page in ["实验报告", "复核记录", "项目说明", "检测工作台"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
