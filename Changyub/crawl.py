"""Python 공식 튜토리얼 5개를 출처·수집 시각과 함께 저장한다."""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup, Tag

ROOT = Path(__file__).resolve().parent
BASE_URL = "https://docs.python.org/ko/3.13/"
USER_AGENT = "ChangyubCourseRAG/1.0 (educational; five tutorial pages)"
PAGES = ["introduction", "controlflow", "datastructures", "modules", "errors"]


def sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def extract_sections(html, page_url):
    """Sphinx 본문만 읽고 제목, 문단, 코드 블록의 순서를 보존한다."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one("div.body") or soup.select_one('main')
    if body is None:
        raise ValueError(f"본문을 찾지 못했습니다: {page_url}")
    for node in body.select("a.headerlink, script, style"):
        node.decompose()
    sections = []
    current = None
    blocks = {"p", "pre", "li", "dt", "dd"}
    for node in body.find_all(["h1", "h2", "h3", "h4", *sorted(blocks)]):
        # li > p 또는 dd > pre 같은 중첩 블록을 중복 추출하지 않는다.
        if any(isinstance(p, Tag) and p.name in blocks for p in node.parents):
            continue
        text = node.get_text("" if node.name == "pre" else " ", strip=node.name != "pre").strip()
        if not text:
            continue
        if node.name.startswith("h"):
            ancestor = node.find_parent("section")
            fragment = node.get("id") or (ancestor.get("id", "") if ancestor else "")
            current = {"heading": text, "url": page_url + ("#" + fragment if fragment else ""), "parts": []}
            sections.append(current)
        else:
            if current is None:
                current = {"heading": "본문", "url": page_url, "parts": []}
                sections.append(current)
            current["parts"].append(text)
    return [dict(heading=s["heading"], url=s["url"], text="\n\n".join(s["parts"]))
            for s in sections if s["parts"]]


def crawl():
    out = ROOT / "source_data"
    raw_dir = out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    robots_url = "https://docs.python.org/robots.txt"
    response = session.get(robots_url, timeout=30)
    response.raise_for_status()
    robots = RobotFileParser(robots_url)
    robots.parse(response.text.splitlines())
    (raw_dir / "robots.txt").write_text(response.text, encoding="utf-8")

    def fetch(url):
        if not robots.can_fetch(USER_AGENT, url):
            raise RuntimeError(f"robots.txt에서 수집을 허용하지 않습니다: {url}")
        time.sleep(max(0.5, robots.crawl_delay(USER_AGENT) or 0))
        res = session.get(url, timeout=30)
        res.raise_for_status()
        if urlparse(res.url).hostname != "docs.python.org":
            raise RuntimeError("예상하지 않은 사이트로 리디렉션되었습니다.")
        if "text/html" not in res.headers.get("Content-Type", ""):
            raise RuntimeError(f"HTML이 아닌 응답입니다: {url}")
        return res.content.decode("utf-8")

    manifest = []
    for name in PAGES:
        url = BASE_URL + f"tutorial/{name}.html"
        html = fetch(url)
        sections = extract_sections(html, url)
        text = "\n\n".join(s["heading"] + "\n" + s["text"] for s in sections)
        if len(text) < 500:
            raise RuntimeError(f"수집 본문이 너무 짧습니다: {url}")
        stamp = datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
        title = BeautifulSoup(html, "html.parser").title.get_text(strip=True)
        doc = dict(id=name, title=title, url=url, crawled_at=stamp,
                   sha256=sha256(text), html_sha256=sha256(html), chars=len(text),
                   section_count=len(sections), sections=sections)
        (raw_dir / f"{name}.html").write_text(html, encoding="utf-8")
        (out / f"{name}.txt").write_text(text + "\n", encoding="utf-8")
        (out / f"{name}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({k: v for k, v in doc.items() if k != "sections"})
        print(f"수집: {name} | {len(text):,}자 | {len(sections)}개 절", flush=True)
    for name in ["license", "copyright"]:
        html = fetch(BASE_URL + name + ".html")
        sections = extract_sections(html, BASE_URL + name + ".html")
        (out / f"{name.upper()}_PYTHON.txt").write_text(
            "Source: " + BASE_URL + name + ".html\n\n" +
            "\n\n".join(s["heading"] + "\n" + s["text"] for s in sections), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    crawl()
