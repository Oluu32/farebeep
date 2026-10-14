"""Jinja2 rendering for the HTML pages (templates in FareBeep/web/templates/).

Autoescaping is on for .html templates, so user-supplied values (names,
emails, tokens) are escaped by default. Pages pass plain values in; the
templates decide markup.
"""
from pathlib import Path

from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATE_DIR = Path(__file__).resolve().parent / "web" / "templates"

_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html"]),
)


def render_html(name: str, status_code: int = 200, **context) -> HTMLResponse:
    """Render web/templates/<name> with the given context as an HTMLResponse."""
    html = _env.get_template(name).render(**context)
    return HTMLResponse(html, status_code=status_code)
