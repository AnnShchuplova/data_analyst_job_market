import sys
import os
from streamlit.web import cli as stcli


def resolve_path(path):
    if getattr(sys, "frozen", False):
        basedir = sys._MEIPASS
    else:
        basedir = os.path.dirname(__file__)
    return os.path.join(basedir, path)


if __name__ == "__main__":
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    main_script = resolve_path(os.path.join("app", "main.py"))
    sys.argv = [
        "streamlit",
        "run",
        main_script,
        "--global.developmentMode=false",
    ]

    sys.exit(stcli.main())