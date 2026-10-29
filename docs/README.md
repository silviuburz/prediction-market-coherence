# Paper landing page

This directory is a static project page for the WC-coherence article.
It links to the IEEE version of record and the public replication repository.
It does not contain another manuscript PDF, a raw-data download, or a backend.

**Website:** [silviuburz.github.io/prediction-market-coherence](https://silviuburz.github.io/prediction-market-coherence/)

## Local preview

From the repository root, run:

```powershell
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8765/`. Stop the server with Ctrl+C.
There is no package installation or website build step.

The page and all links work without JavaScript. JavaScript adds the Copy
BibTeX button and reports clipboard failures on the page. The citation can
also be selected or downloaded.

## Files and checks

- [index.html](./index.html): title, authors, abstract, results and resource links.
- [styles.css](./styles.css): desktop, mobile and print layout.
- [site.js](./site.js): clipboard action only; no tracking or remote scripts.
- [assets/instrument.svg](./assets/instrument.svg): the authors' Figure 1,
  exported from the original vector PDF with text converted to paths.
- [citation.bib](./citation.bib): the associated article citation.

The journal metadata was checked against the public Crossref DOI record
on 29 October 2026. The DOI now resolves to IEEE Xplore. Interim page numbers
from the metadata record are not included in the citation.

Run the site checks from the repository root:

```powershell
python code\test_site.py
```

The checks bind visible result values to the supplied analysis artifacts,
check citation consistency, and resolve local page links. Existing numerical
auditors remain unchanged.

## GitHub Pages

The publishing source is **Deploy from a branch**, branch **main**,
folder **/docs**. Approved changes go live after they are committed, pushed
and deployed by GitHub Pages.
The [GitHub instructions](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)
describe that process.

Relative asset links work at this project address and in the local preview.
The `.nojekyll` file keeps the files as a plain static site.
