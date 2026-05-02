project = "xharmonics"
author = "OpenAI Codex"
extensions = ["myst_nb", "myst_parser"]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
    ".ipynb": "myst-nb",
}
master_doc = "index"
exclude_patterns = ["_build"]
html_theme = "alabaster"
nb_execution_mode = "off"
