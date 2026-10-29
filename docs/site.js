"use strict";

const copyButton = document.getElementById("copy-citation");
const citation = document.getElementById("bibtex");
const copyStatus = document.getElementById("copy-status");

copyButton.hidden = false;
copyButton.addEventListener("click", async () => {
  copyStatus.classList.remove("error");
  if (!navigator.clipboard || !navigator.clipboard.writeText) {
    copyStatus.classList.add("error");
    copyStatus.textContent = "Clipboard access is unavailable. Select the citation below or download the .bib file.";
    return;
  }
  try {
    await navigator.clipboard.writeText(citation.textContent.trim() + "\n");
    copyStatus.textContent = "BibTeX copied.";
  } catch {
    copyStatus.classList.add("error");
    copyStatus.textContent = "Copy did not succeed. Select the citation below or download the .bib file.";
  }
});
