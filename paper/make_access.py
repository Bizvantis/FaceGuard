"""Build the IEEE Access version of the paper: one self-contained .tex for the official IEEE Access template.

    python make_access.py BUILD_DIR OUT.tex

BUILD_DIR must hold faceguard.bbl from a LaTeX/BibTeX run of faceguard.tex (so the reference list is identical
to the IEEEtran version). The output needs only the IEEE Access class files (ieeeaccess.cls, spotcolor.sty,
logo.png, notaglinelogo.png, bullet.png - all in IEEE's template) and the figures/ folder:

* every number macro of generated/numbers.tex and every generated table is written out in full,
* the front matter uses the template's \\history, \\doi, \\address, \\tfootnote, \\corresp, abstract and keywords,
* the abstract avoids abbreviations, as the IEEE Access instructions require,
* the formatted reference list is embedded (thebibliography, as in the template), so no .bib file is needed,
* author biographies (to be completed) and \\EOD close the article, as in the template.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PREAMBLE = r"""\documentclass{ieeeaccess}
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
% ieeeaccess.cls redefines \year as a command, but TikZ reads it as a number while loading: lend TikZ the TeX
% primitive and restore the class's \year afterwards.
\let\IEEEaccessyear\year
\ExplSyntaxOn \cs_gset_eq:NN \year \tex_year:D \ExplSyntaxOff
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc,shapes.geometric,backgrounds}
\let\year\IEEEaccessyear
\usepackage{url}
\usepackage{flafter}

\usepackage{bm}
\makeatletter
\AtBeginDocument{\DeclareMathVersion{bold}
\SetSymbolFont{operators}{bold}{T1}{times}{b}{n}
\ifcsname symNewLetters\endcsname\SetSymbolFont{NewLetters}{bold}{T1}{times}{b}{it}\fi
\SetMathAlphabet{\mathrm}{bold}{T1}{times}{b}{n}
\SetMathAlphabet{\mathit}{bold}{T1}{times}{b}{it}
\SetMathAlphabet{\mathbf}{bold}{T1}{times}{b}{n}
\SetMathAlphabet{\mathtt}{bold}{OT1}{pcr}{b}{n}
\SetSymbolFont{symbols}{bold}{OMS}{cmsy}{b}{n}
\renewcommand\boldmath{\@nomath\boldmath\mathversion{bold}}}
\makeatother

% The class's caption code tests \xfigwd, which only its \Figure macro sets; standard figure environments need a
% value (0pt = captions wrap within the figure width).
\providecommand{\xfigwd}{0pt}

\newtheorem{proposition}{Proposition}
\algnewcommand{\IfThen}[2]{\State \algorithmicif\ #1\ \algorithmicthen\ #2} % one-line "if ... then ..."
\definecolor{fgblue}{HTML}{2A78D6}
\definecolor{fgorange}{HTML}{EB6834}
\definecolor{fgink}{HTML}{0B0B0B}
\definecolor{fgmuted}{HTML}{898781}
\definecolor{fgfill}{HTML}{EEF4FC}
\definecolor{fggrey}{HTML}{F4F3F0}

\hyphenation{Face-Guard Media-Pipe web-cam web-cams op-er-at-ing}

%Your document starts from here ___________________________________________________
\begin{document}
\history{Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.}
\doi{10.1109/ACCESS.2024.0429000}

\title{FaceGuard: Continuous Face Authentication With a Homography-Parallax Liveness Cue for Commodity Webcams}
\author{\uppercase{First A. Author}\authorrefmark{1}, \IEEEmembership{Student Member, IEEE},
AND \uppercase{Second B. Author}\authorrefmark{1}}

\address[1]{Department of Computer Science and Engineering, Institution Name, City, Country (e-mail: author@example.edu)}
\tfootnote{The source code and experiment scripts are available at \url{https://github.com/Bizvantis/FaceGuard}.}

\markboth
{Author \headeretal: FaceGuard: Continuous Face Authentication With a Homography-Parallax Liveness Cue}
{Author \headeretal: FaceGuard: Continuous Face Authentication With a Homography-Parallax Liveness Cue}

\corresp{Corresponding author: First A. Author (e-mail: author@example.edu).}

"""

BIOGRAPHIES = r"""
\begin{IEEEbiographynophoto}{First A. Author} (Student Member, IEEE) [Degree, field, institution, city, country, year.
Current position and research interests. IEEE Access asks for a photograph of 1 in $\times$ 1.25 in with the
final files; then use the IEEEbiography environment of the template.]
\end{IEEEbiographynophoto}

\begin{IEEEbiographynophoto}{Second B. Author} [Degrees, current position, affiliation and research interests.]
\end{IEEEbiographynophoto}

\EOD

\end{document}
"""


def brace_arg(s: str, i: int) -> tuple[str, int]:
    """Return the {...} group starting at s[i] == '{' and the index after it."""
    assert s[i] == "{", s[i:i + 30]
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\":
            depth += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def load_macros() -> dict[str, str]:
    macros = {}
    for f in sorted((HERE / "generated").glob("*.tex")):
        s = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\\newcommand\{\\([A-Za-z]+)\}", s):
            body, _ = brace_arg(s, m.end())
            macros[m.group(1)] = body[1:].lstrip("\n") if body.startswith("%") else body
    return macros


def expand(s: str, macros: dict[str, str]) -> str:
    names = sorted(macros, key=len, reverse=True)
    pat = re.compile(r"\\(" + "|".join(names) + r")(?![A-Za-z])(\{\})?([ \t]*)")

    def sub(m):
        val = macros[m.group(1)]
        # TeX gobbles the spaces after a control word unless the word is closed by {} - emulate that
        return val + (m.group(3) if m.group(2) else "")

    for _ in range(5):  # row macros contain number macros
        s2 = pat.sub(sub, s)
        if s2 == s:
            break
        s = s2
    return s


def access_body(build: Path) -> tuple[str, str]:
    """(abstract, abstract/keywords/\\maketitle + main text + embedded references) in IEEE Access form."""
    src = (HERE / "faceguard.tex").read_text(encoding="utf-8")
    body = src[src.index(r"\begin{abstract}"):]
    abstract, rest = body.split(r"\end{abstract}", 1)
    abstract = abstract.replace(r"\begin{abstract}", "").strip()
    kw = rest[rest.index(r"\begin{IEEEkeywords}") + len(r"\begin{IEEEkeywords}"):rest.index(r"\end{IEEEkeywords}")].strip()
    terms = sorted((t.strip().rstrip(".") for t in kw.split(",")), key=str.lower)  # IEEE Access: alphabetical order
    kw = ", ".join([terms[0][0].upper() + terms[0][1:]] + [t[0].lower() + t[1:] for t in terms[1:]]) + "."
    main_text = rest[rest.index(r"\end{IEEEkeywords}") + len(r"\end{IEEEkeywords}"):]
    main_text = main_text[:main_text.index(r"\bibliographystyle")]

    # IEEE Access: the abstract must be self-contained and without abbreviations
    for a, b in [("an ordinary RGB webcam", "an ordinary color webcam"),
                 (r"\sys{}, a CPU-only system that adds", r"\sys{}, a system that runs on an ordinary laptop processor and adds"),
                 ("while leaving password or PIN sign-in untouched", "while leaving password sign-in untouched")]:
        assert abstract.count(a) == 1, a
        abstract = abstract.replace(a, b)

    text = ("\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n\\begin{keywords}\n" + kw +
            "\n\\end{keywords}\n\n\\titlepgskip=-21pt\n\n\\maketitle\n" + main_text)
    text = text.replace(r"\IEEEPARstart", r"\PARstart")
    # IEEE Access columns are 85.3 mm instead of 88.9 mm: the open-set table needs tighter column spacing
    a = "\\label{tab:openset}\n\\footnotesize\n\\setlength{\\tabcolsep}{1.6pt}"
    assert text.count(a) == 1
    text = text.replace(a, a.replace("1.6pt", "1.2pt"))
    # long file and model names in typewriter type may break after an underscore (the columns are narrower)
    text = re.sub(r"\\texttt\{([^{}]{16,})\}", lambda m: r"\texttt{" + m.group(1).replace(r"\_", r"\_\allowbreak{}") + "}", text)
    text = expand(text, load_macros())
    text = text.replace(r"\sys{}", "FaceGuard").replace(r"\sys", "FaceGuard")
    left = sorted(set(re.findall(r"\\(?:lfw|rw|par|att|liv|bs|db|ms|ar|thr|nz|pr|pd|mp|rob|os|id|cls|g|ad|rt|op|sens|lb|eff|p)[A-Z][A-Za-z]*", text)))
    if left:
        print("WARNING: unexpanded macros:", left)

    bbl = (build / "faceguard.bbl").read_text(encoding="utf-8").strip()
    return abstract, text.rstrip() + "\n\n" + bbl + "\n"


def main(build: Path, out: Path) -> None:
    abstract, body = access_body(build)
    out.write_text(PREAMBLE + body + BIOGRAPHIES, encoding="utf-8")
    words = len(re.sub(r"\\[a-zA-Z]+|[{}$\\]", " ", abstract).split())
    print(f"[access] {out}  (abstract ~{words} words)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
