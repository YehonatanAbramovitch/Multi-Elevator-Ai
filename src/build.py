"""Build index.html from src/lab_template.html, src/core.js and data/.

    python src/build.py
"""
import json
import pathlib

root = pathlib.Path(__file__).resolve().parent.parent
template = (root / "src" / "lab_template.html").read_text(encoding="utf-8")
core = (root / "src" / "core.js").read_text(encoding="utf-8")
problems = (root / "data" / "problems.json").read_text(encoding="utf-8")
ai_final = json.dumps(json.loads((root / "data" / "ai_final.json").read_text()), separators=(",", ":"))
for name, text in (("core.js", core), ("problems.json", problems)):
    assert "</script" not in text.lower(), f"{name} must not contain </script"

page = template.replace("__CORE__", core).replace("__PROBLEMS__", problems).replace("__AIFINAL__", ai_final)
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n</head>\n<body>\n')
(root / "index.html").write_text(head + page + "\n</body>\n</html>\n", encoding="utf-8")
print("wrote index.html")
