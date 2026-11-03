#!/usr/bin/env python3
"""Gate for the GPU kernel knowledge base. Run before every commit.

Checks:
- broken relative links in .md and .html files
- note frontmatter: parses, name == filename stem, non-empty description and source
- notes missing from every README
- skills: skills/<name>/SKILL.md has name == <name>, a non-empty description,
  and is linked from a README
- leaks: personal paths, internal hosts, container names, ticket ids, emails,
  credential-like strings

Exit code 1 if anything fails.

usage: python3 skills/kb-add-topic/scripts/check_kb.py [kb_root]
       (default kb_root: the repository that contains this script)
"""
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
ROOT = os.path.abspath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ROOT))
SKIP_DIRS = {".git"}
errors = []

LEAK_PATTERNS = [
    (r"/home/[a-z0-9_-]+|/Users/[A-Za-z0-9_-]+", "personal path"),
    (r"\b[a-z0-9-]+\.amd\.com\b", "internal host (allowed: www/docs/rocm.docs/developer)"),
    (r"docker exec \S+", "container name"),
    (r"\b(SILOTIGER|ROCM|JIRA|PLAT|SCET)-\d+\b", "ticket id"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "email"),
    (r"(api[_-]?key|token|secret|password)\s*[:=]", "credential-like"),
]
ALLOWED_HOSTS = re.compile(r"^(www|docs|rocm\.docs|developer)\.amd\.com$")
ALLOWED_EMAILS = re.compile(r"noreply")


def walk():
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in SKIP_DIRS]
        for f in fns:
            yield os.path.join(dp, f)


def rel(p):
    return os.path.relpath(p, ROOT)


def frontmatter(p):
    """Return the parsed frontmatter dict, or None after recording an error."""
    t = open(p, encoding="utf-8").read()
    if not t.startswith("---\n"):
        errors.append(f"no frontmatter  {rel(p)}")
        return None
    try:
        return yaml.safe_load(t[4:].split("\n---\n", 1)[0]) or {}
    except Exception as e:
        errors.append(f"bad YAML  {rel(p)}: {str(e).splitlines()[0]} (quote the value)")
        return None


def is_skill_tree(p):
    return rel(p).startswith("skills" + os.sep)


files = list(walk())
notes = [p for p in files if p.endswith(".md") and os.path.basename(p) != "README.md"
         and not rel(p).startswith("_sources") and not is_skill_tree(p)]
skills = [p for p in files if os.path.basename(p) == "SKILL.md" and is_skill_tree(p)]
readmes = [p for p in files if os.path.basename(p) == "README.md"]

# 1. links + leaks
# Links inside code (fenced blocks and inline `spans`) are examples, not links, so
# only prose is link-checked. Leak checks scan every line, examples included.
linked = set()
for p in files:
    if not p.endswith((".md", ".html")):
        continue
    in_fence = False
    for i, line in enumerate(open(p, encoding="utf-8"), 1):
        is_fence = line.lstrip().startswith("```")
        if is_fence:
            in_fence = not in_fence
        prose = "" if (in_fence or is_fence) else re.sub(r"`[^`]*`", "", line)
        for m in re.finditer(r"\]\(([^)\s]+)\)", prose):
            t = m.group(1)
            if t.startswith(("http", "mailto:", "#")):
                continue
            path = os.path.normpath(os.path.join(os.path.dirname(p), t.split("#")[0]))
            if t.split("#")[0] and not os.path.exists(path):
                errors.append(f"broken link  {rel(p)}:{i}: {t}")
            elif os.path.basename(p) == "README.md":
                linked.add(path)
        for pat, what in LEAK_PATTERNS:
            for m in re.finditer(pat, line, re.I):
                s = m.group(0)
                if what.startswith("internal host") and ALLOWED_HOSTS.match(s.lower()):
                    continue
                if what == "email" and ALLOWED_EMAILS.search(s):
                    continue
                errors.append(f"leak ({what})  {rel(p)}:{i}: {s}")

# 2. note frontmatter
for p in notes:
    fm = frontmatter(p)
    if fm is None:
        continue
    for k in ("name", "description", "source"):
        if not fm.get(k):
            errors.append(f"empty {k}  {rel(p)}")
    if fm.get("name") != os.path.splitext(os.path.basename(p))[0]:
        errors.append(f"name != filename  {rel(p)}: {fm.get('name')}")

# 3. every note reachable from some README
for p in notes:
    if p not in linked:
        errors.append(f"not indexed  {rel(p)}")

# 4. skills: name == folder, description set, linked from a README
for p in skills:
    fm = frontmatter(p)
    if fm is None:
        continue
    folder = os.path.basename(os.path.dirname(p))
    if fm.get("name") != folder:
        errors.append(f"skill name != folder  {rel(p)}: {fm.get('name')}")
    if not fm.get("description"):
        errors.append(f"empty description  {rel(p)}")
    if p not in linked:
        errors.append(f"skill not indexed  {rel(p)} (link it from skills/README.md)")

print("\n".join(errors) if errors else "OK")
print(f"{len(files)} files, {len(notes)} notes, {len(skills)} skills, {len(errors)} problems",
      file=sys.stderr)
sys.exit(1 if errors else 0)
