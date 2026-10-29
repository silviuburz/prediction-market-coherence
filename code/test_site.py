"""Check the paper site's links, citation, vector figure and numerical claims."""
import json
import re
import unittest
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"}


class PaperPage(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = set()
        self.links = []
        self.stack = []
        self.values = {}
        self.meta = {}
        self.images = []
        self.scripts = []
        self.visible = []

    def handle_starttag(self, tag, attributes):
        attributes = dict(attributes)
        identifier = attributes.get("id")
        if identifier:
            if identifier in self.ids:
                raise ValueError(f"Duplicate page ID: {identifier}")
            self.ids.add(identifier)
        for key in ("href", "src"):
            if key in attributes:
                self.links.append(attributes[key])
        if tag == "meta" and "name" in attributes:
            self.meta.setdefault(attributes["name"], []).append(attributes["content"])
        if tag == "img":
            self.images.append(attributes)
        if tag == "script":
            self.scripts.append(attributes)
        if tag not in VOID:
            field = attributes.get("data-claim") or identifier
            self.stack.append((tag, field))
            if field:
                self.values[field] = ""

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1][0] != tag:
            raise ValueError(f"Unexpected closing tag: {tag}")
        self.stack.pop()

    def handle_data(self, data):
        for _tag, field in self.stack:
            if field:
                self.values[field] += data
        if not any(tag in ("script", "style", "head") for tag, _field in self.stack):
            self.visible.append(data)


def read_page():
    page = PaperPage()
    page.feed((SITE / "index.html").read_text(encoding="utf-8"))
    page.close()
    if page.stack:
        raise ValueError("Unclosed HTML elements")
    return page


class PaperSiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = read_page()

    def test_title_authors_and_doi_match_citation_metadata(self):
        cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        preferred = cff.split("preferred-citation:", 1)[1]
        title = re.search(r'(?m)^  title: "([^"]+)"$', preferred)[1]
        doi = re.search(r'(?m)^  doi: "([^"]+)"$', preferred)[1]
        self.assertEqual(self.page.values["paper-title"], title)
        self.assertEqual(self.page.meta["citation_title"], [title])
        self.assertEqual(self.page.meta["citation_doi"], [doi])
        self.assertEqual(self.page.meta["citation_author"],
                         ["Burz, Silviu", "Chiru, Costin-Gabriel"])
        self.assertIn("https://doi.org/" + doi, self.page.links)

    def test_visible_results_match_the_supplied_artifacts(self):
        data = ROOT / "figures" / "data"
        corrected = json.loads((data / "corrected_specification.json").read_text())
        corrected = next(row for row in corrected["runs"] if row["cohort"] == "holdout")
        original = json.loads((data / "event_study_holdout.json").read_text())["isolated"]
        provenance = json.loads((ROOT / "PROVENANCE.json").read_text())
        expected = {
            "corrected-effect": f"{corrected['D_cents']:+.3f}",
            "corrected-low": f"{corrected['ci_cents'][0]:.3f}",
            "corrected-high": f"{corrected['ci_cents'][1]:.3f}",
            "evaluation-goals": str(corrected["n_events"]),
            "evaluation-matches": str(corrected["n_matches"]),
            "positive-matches": str(corrected["n_pos"]),
            "original-effect": f"{100 * original['D']:+.3f}",
            "corpus-files": f"{provenance['raw_corpus']['n_files']:,}",
            "corpus-matches": str(provenance["split"]["n_dev"]
                                  + provenance["split"]["n_holdout"]),
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                self.assertEqual(self.page.values[field].strip(), value)
        visible = " ".join(" ".join(self.page.visible).split())
        self.assertIn("not a second confirmation", visible)
        self.assertIn("23.70 GB raw corpus is not included", visible)
        self.assertIn("not executable profit", visible)

    def test_local_links_and_page_anchors_resolve(self):
        for link in self.page.links:
            parsed = urlsplit(link)
            if parsed.scheme:
                self.assertEqual(parsed.scheme, "https", link)
                continue
            self.assertFalse(parsed.netloc, link)
            self.assertFalse(parsed.path.startswith("/"), link)
            if parsed.path:
                path = (SITE / unquote(parsed.path)).resolve()
                self.assertTrue(path.is_relative_to(SITE), link)
                self.assertTrue(path.is_file(), link)
            else:
                self.assertIn(parsed.fragment, self.page.ids, link)

    def test_bibtex_download_and_display_are_identical(self):
        citation = (SITE / "citation.bib").read_text(encoding="utf-8").strip()
        self.assertEqual(self.page.values["bibtex"].strip(), citation)
        self.assertIn("{Burz, Silviu and Chiru, Costin-Gabriel}", citation)
        self.assertIn("{10.1109/ACCESS.2026.3746471}", citation)
        self.assertNotIn("pages", citation)
        self.assertNotIn("volume", citation)

    def test_figure_is_accessible_vector_artwork(self):
        self.assertEqual(len(self.page.images), 1)
        image = self.page.images[0]
        self.assertGreater(len(image["alt"]), 80)
        svg = ET.parse(SITE / image["src"]).getroot()
        namespace = "{http://www.w3.org/2000/svg}"
        self.assertFalse(svg.findall(".//" + namespace + "image"))
        self.assertFalse(svg.findall(".//" + namespace + "script"))
        self.assertGreater(len(svg.findall(".//" + namespace + "path")), 100)
        for element in svg.iter():
            for key, value in element.attrib.items():
                if key.rsplit("}", 1)[-1] == "href":
                    self.assertTrue(value.startswith("#"), value)

    def test_site_has_no_external_runtime_dependencies(self):
        self.assertEqual(self.page.scripts, [{"src": "site.js", "defer": None}])
        css = (SITE / "styles.css").read_text(encoding="utf-8")
        self.assertNotIn("@import", css)
        self.assertNotIn("url(", css)
        self.assertIn("prefers-reduced-motion", css)
        self.assertTrue((SITE / ".nojekyll").is_file())
        self.assertIn('href="#main"', (SITE / "index.html").read_text())


if __name__ == "__main__":
    unittest.main()
