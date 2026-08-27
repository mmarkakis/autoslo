"""Generate the code reference pages and navigation.

Script was taken from
https://mkdocstrings.github.io/recipes/#automatic-code-reference-pages
"""

"""Generate the code reference pages and navigation."""

"""Generate the code reference pages."""

from pathlib import Path
import sys
import mkdocs_gen_files


# Replace the file docs/index.md with a copy of README.md
with mkdocs_gen_files.open("index.md", "w") as fd:
    with open("README.md") as readme:
        fd.write(readme.read())


nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "src"
sys.path.append(str(src))  # allow importing from src

# Generate CSS from autoslo.utils.colors.Palette
try:
    from autoslo.visualizations.colors import Palette as _CLPalette

    pal = _CLPalette()
    sem = pal.semantic_colors()

    css = f"""/* Generated from autoslo.utils.colors.Palette */
:root {{
  --cl-white: {pal.white};
  --cl-black: {pal.black};
  --cl-gray: {pal.gray};
  --cl-green-light: {pal.light_green};
  --cl-green-dark: {pal.dark_green};
  --cl-blue-light: {pal.light_blue};
  --cl-blue-dark: {pal.dark_blue};
  --cl-yellow-light: {pal.light_yellow};
  --cl-yellow-dark: {pal.dark_yellow};
  --cl-orange-light: {pal.light_orange};
  --cl-orange-dark: {pal.dark_orange};
  --cl-red-light: {pal.light_red};
  --cl-red-dark: {pal.dark_red};
}}

/* Light scheme (Material: default) */
[data-md-color-scheme="default"] {{
  --md-primary-fg-color: {pal.dark_green};
  --md-primary-fg-color--light: {pal.light_green};
  --md-primary-fg-color--dark: {pal.dark_green};
  --md-accent-fg-color: {pal.dark_orange};
}}

/* Dark scheme (Material: slate) */
[data-md-color-scheme="slate"] {{
  --md-primary-fg-color: {pal.light_green};
  --md-primary-fg-color--light: {pal.light_green};
  --md-primary-fg-color--dark: {pal.dark_green};
  --md-accent-fg-color: {pal.light_orange};
}}

/* Admonitions accents */
.md-typeset .admonition.tip, .md-typeset details.tip {{ border-color: {sem.get("success", pal.dark_green)}; }}
.md-typeset .admonition.info, .md-typeset details.info {{ border-color: {sem.get("info", pal.dark_blue)}; }}
.md-typeset .admonition.warning, .md-typeset details.warning {{ border-color: {sem.get("warning", pal.dark_yellow)}; }}
.md-typeset .admonition.danger, .md-typeset details.danger {{ border-color: {sem.get("error", pal.dark_red)}; }}
"""
    with mkdocs_gen_files.open("assets/generated/palette.css", "w") as fd:
        fd.write(css)
except Exception:
    # Keep docs build resilient if runtime deps are missing
    pass

# New: recurse under 'autoslo' and skip FastAPI runtime modules ('autoslo/api/*')
pkg_root = src / "autoslo"
for path in sorted(pkg_root.rglob("*.py")):
    rel = path.relative_to(src)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "autoslo" and parts[1] == "api":
        continue

    module_path = path.relative_to(src).with_suffix("")
    doc_path = path.relative_to(src).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    print(doc_path.as_posix())

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        print("::: " + identifier, file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))


with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
