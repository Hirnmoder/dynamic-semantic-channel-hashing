import itables
from itables import show as idf

import dsh.utils.logger

dsh.utils.logger.Logger(dsh.utils.logger.ConsoleLoggerTarget(), minimum_log_level=dsh.utils.logger.LogLevel.WARNING)

__all__ = ["idf", "RenderJSON"]

itables.init_notebook_mode(all_interactive=False, connected=False)
itables.options.column_filters = False
itables.options.lengthMenu = [5, 12, 25, 100, 1000]
itables.options.pageLength = 12
itables.options.buttons = ["pageLength", "copyHtml5", dict(extend="csvHtml5", title="data")]
itables.options.columnControl = [  # type: ignore
    "order",
    [
        "search",
        "spacer",
        "orderAsc",
        "orderDesc",
        "spacer",
        "orderAddAsc",
        "orderAddDesc",
        "orderRemove",
        "orderClear",
        "spacer",
        dict(extend="dropdown", text="Column Visibility", content=["colVis"]),
    ],
]
itables.options.columnDefs = [dict(targets="_all", className="dt-head-left")]
itables.options.ordering = {"indicators": False, "handler": False}  # type: ignore
itables.options.search = {"regex": True, "caseInsensitive": True}


# The following is based on https://stackoverflow.com/a/37124230
import uuid
from IPython.display import display_javascript, display_html


class RenderJSON(object):
    def __init__(self, name: str, json_str: str):
        self.name = name
        self.json_str = json_str
        self.uuid = str(uuid.uuid4())

    def _ipython_display_(self):
        display_html(
            f'<div id="{self.uuid}" style="width:100%;"><span style="font-family:monospace; margin-bottom:-20px;">{self.name}</span></div>',
            raw=True,
        )
        display_javascript(
            """
                require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
                    document.getElementById('%s').appendChild(renderjson(%s))
                });
            """
            % (self.uuid, self.json_str),
            raw=True,
        )
