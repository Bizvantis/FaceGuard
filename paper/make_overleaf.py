"""Build the Overleaf project of the paper in the layout of the IEEE Access reference project (Sriram-suthir.zip).

    python make_overleaf.py BUILD_DIR REFERENCE_DIR OUT_DIR

BUILD_DIR      holds faceguard.bbl from a LaTeX/BibTeX run of faceguard.tex
REFERENCE_DIR  the unpacked reference project (ieeeaccess.cls, IEEEtran.cls, logos, bullet.png, orcid_id_image.jpg)
OUT_DIR        receives access.tex, the class and logo files, figures/ and <OUT_DIR>.zip

The text is the IEEE Access version of make_access.py (same content). To keep Overleaf compiles short (the free plan
stops after 10 s): the two TikZ diagrams are pre-rendered to PDF, the interface screenshots are stored as JPEG, the
reference list is embedded, and nothing is generated at compile time.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2

from make_access import access_body

HERE = Path(__file__).resolve().parent
TECTONIC = HERE.parent / "tools" / "tectonic.exe"

# figure files in the order of the figures in the paper
FIGURES = {
    "figures/fig_parallax.pdf": "figures/fig02_yaw_and_parallax_simulation.pdf",
    "figures/screen_live_crop.png": "figures/fig04a_interface_live.jpg",
    "figures/screen_unlock_crop.png": "figures/fig04b_interface_unlock.jpg",
    "figures/fig_roc.pdf": "figures/fig05_roc_curves.pdf",
    "figures/fig_scores.pdf": "figures/fig06_score_distributions.pdf",
    "figures/fig_openset.pdf": "figures/fig07_open_set_identification.pdf",
    "figures/fig_robust.pdf": "figures/fig08_robustness.pdf",
    "figures/fig_parallax_real.pdf": "figures/fig09_parallax_real_vs_simulated.pdf",
    "figures/fig_liveness.pdf": "figures/fig10_live_user_rejection.pdf",
    "figures/fig_runtime.pdf": "figures/fig11_runtime.pdf",
}
TIKZ = ["figures/fig01_architecture.pdf", "figures/fig03_presence_guard_states.pdf"]

PREAMBLE = r"""\documentclass{ieeeaccess}

%=========================================================
% PACKAGES
%=========================================================
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{amsthm}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{subcaption}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}
\usepackage{threeparttable}
\usepackage[hidelinks]{hyperref}

%=========================================================
% FLOAT PLACEMENT & SPACING OPTIMIZATIONS
%=========================================================
\usepackage{placeins}
\usepackage{dblfloatfix} % Fixes 2-column float ordering & placement
\usepackage{caption}

% Tight, professional float spacing
\setlength{\textfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\floatsep}{8pt plus 2pt minus 2pt}
\setlength{\intextsep}{8pt plus 2pt minus 2pt}
\setlength{\dbltextfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\dblfloatsep}{8pt plus 2pt minus 2pt}
\setlength{\abovecaptionskip}{3pt}
\setlength{\belowcaptionskip}{3pt}

% Allow aggressive float packing to eliminate white gaps
\renewcommand{\topfraction}{0.99}
\renewcommand{\bottomfraction}{0.99}
\renewcommand{\textfraction}{0.01}
\renewcommand{\floatpagefraction}{0.70}
\renewcommand{\dbltopfraction}{0.99}
\renewcommand{\dblfloatpagefraction}{0.70}

\setcounter{topnumber}{6}
\setcounter{bottomnumber}{6}
\setcounter{totalnumber}{10}
\setcounter{dbltopnumber}{6}

\clubpenalty=10000
\widowpenalty=10000
\displaywidowpenalty=10000

\captionsetup{belowskip=2pt, aboveskip=2pt, font=small}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

%=========================================================
% DEFINITIONS USED BY THIS PAPER
%=========================================================
\providecommand{\xfigwd}{0pt} % read by the class's caption code
\makeatletter % biography environments of ieeeaccess.cls (needed with some IEEEtran.cls versions)
\ifcsname if@biographyTOCentrynotmade\endcsname\else
  \newif\if@biographyTOCentrynotmade \@biographyTOCentrynotmadetrue
\fi
\@ifundefined{c@biography}{\newcounter{biography}}{}
\makeatother
\newtheorem{proposition}{Proposition}
\algnewcommand{\IfThen}[2]{\State \algorithmicif\ #1\ \algorithmicthen\ #2} % one-line "if ... then ..."
\hyphenation{Face-Guard Media-Pipe web-cam web-cams op-er-at-ing}

\begin{document}

\newcommand{\orcidicon}{%
  \includegraphics[width=0.32cm]{orcid_id_image.jpg}}

\doi{}

%=========================================================
% TITLE
%=========================================================
\title{FaceGuard: Continuous Face Authentication With a Homography-Parallax Liveness Cue for Commodity Webcams}

\author{%
\uppercase{Suthir Sriram}
\href{https://orcid.org/0000-0003-2480-3273}{\orcidicon}
\authorrefmark{1},
\uppercase{Srilekha Totakuri}
\href{https://orcid.org/0000-0002-0134-306X}{\orcidicon}
\authorrefmark{2},
\uppercase{Nivethitha Vijayaraj}
\href{https://orcid.org/0000-0002-8694-033X}{\orcidicon}
\authorrefmark{1},
\uppercase{Pranaysiddarth Jagarlamudi}
\href{https://orcid.org/0009-0001-0581-5861}{\orcidicon}
\authorrefmark{1},
\uppercase{Gaurav Choudhary}
\href{https://orcid.org/0000-0003-3378-2945}{\orcidicon}
\authorrefmark{3}}

\address[1]{Department of Computer Science and Engineering,
Amrita School of Computing, Amrita Vishwa Vidyapeetham,
Chennai 600103, India}
\address[2]{Vaagdevi Engineering College, Warangal,
Telangana, India}
\address[3]{Section of CyberSecurity Engineering, DTU Compute,
Technical University of Denmark (DTU),
2800 Kongens Lyngby, Denmark}

\markboth
{Suthir Sriram \headeretal: FaceGuard: Continuous Face Authentication
With a Homography-Parallax Liveness Cue}
{Suthir Sriram \headeretal: FaceGuard: Continuous Face Authentication
With a Homography-Parallax Liveness Cue}

\corresp{Corresponding author: Suthir Sriram
(e-mail: s\_suthir@ch.amrita.edu)}

\tfootnote{This research was carried out using the facilities and support of
the NEGCES Lab, Amrita Vishwa Vidyapeetham, Chennai Campus.}

%=========================================================
% ABSTRACT
%=========================================================
"""

ENDING = r"""
%=========================================================
% AUTHOR BIOGRAPHIES
%=========================================================
\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{figures/author_suthir_sriram.jpg}}]
{\textbf{Suthir Sriram}}
is an Assistant Professor (Selection Grade) in the Department of Computer Science and Engineering at the Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai Campus. He holds a Ph.D. in Computer Science and Engineering from MS University, an M.Tech in Information Technology from Sathyabama University, Chennai, and a B.Tech in Information Technology from Anna University, Chennai. With over 18.5 years of academic experience, he has published more than 100 research papers in reputed international journals. He is currently supervising 6 Ph.D. scholars and has successfully co-supervised 3 Ph.D. scholars to completion. His primary research domains include Computer Vision, Data Science, Machine Learning, and Network Intelligence.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{figures/author_srilekha_totakuri.jpg}}]
{\textbf{Srilekha Totakuri}}
received the B.E. degree in CSE from the AnuBose Institute of Technology, Polvancha, in 2012, the M.E. degree from the Jayamukhi Institute of Technology, Narsampet, Warangal, Telangana, in 2017, and the Ph.D. degree from the Department of Computer Science and Engineering, Annamalai University, in 2024. She is currently an Assistant Professor with the Vaagdevi Engineering College, Bollikunta, Warangal. Her research interests include AI and cloud computing.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{figures/author_nivethitha_vijayaraj.jpg}}]
{\textbf{Nivethitha Vijayaraj}}
is currently pursuing her Ph.D. (Thesis submitted) at Vellore Institute of Technology (VIT), Chennai Campus, and serves as an Assistant Professor in the Department of Computer Science and Engineering at the Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai. She holds an M.E. in Computer Science from an Anna University affiliated college in Chennai and a B.E. in Electronics and Communication Engineering from a Coimbatore Anna University affiliated institution. She has 12 years of teaching experience and 3 years of industry experience, with more than 30 publications in journals and conferences. Her research interests include optical communications, IoT, machine learning, and photonics.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{figures/author_pranaysiddarth_jagarlamudi.jpg}}]
{\textbf{Pranaysiddarth Jagarlamudi}}
is currently pursuing the B.Tech. degree in Computer Science and Engineering at the Amrita School of Computing, Amrita Vishwa Vidyapeetham, Chennai, India. The research work covers the design, implementation, and evaluation of camera-based authentication systems, including the FaceGuard system presented in this article. Research interests include computer vision, biometric security, in particular face authentication and presentation-attack detection, and machine learning and deep learning.
\end{IEEEbiography}

\begin{IEEEbiography}[{\includegraphics[width=1in,height=1.25in,clip,keepaspectratio]{figures/author_gaurav_choudhary.jpg}}]
{\textbf{Gaurav Choudhary}}
is an Associate Professor with the Section for Cybersecurity Engineering, DTU Compute, Technical University of Denmark (DTU), Copenhagen. He has worked on projects funded by Horizon Europe, Industriensfond, the National Cluster Digital Technologies and Innovation, DigitalLead, IITP, NRF Korea, and AFOSR USA. He has authored or co-authored more than 70 peer-reviewed scientific papers in international journals.
\end{IEEEbiography}

\EOD

\end{document}
"""

DATA_AVAILABILITY = r"""%=========================================================
% DATA AND CODE AVAILABILITY
%=========================================================
\section*{Data and Code Availability}
The source code and experiment scripts are available at \url{https://github.com/Bizvantis/FaceGuard}. The repository contains the FaceGuard application with its automated tests, and the experiment scripts that regenerate every figure, table, and reported number of this article together with the result files they produce. The public LFW and MPIIFaceGaze datasets and the pretrained models are not redistributed because of their licenses; the repository documents how to obtain them, and every model is verified against a pinned SHA-256 hash. Raw per-frame logs of the real-world evaluation are withheld to protect the participants' privacy, and their anonymous summary is included.

"""

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def render_tikz(picture: str, out_pdf: Path, guard_section: str) -> None:
    """Compile one tikzpicture into a tightly cropped PDF (same code, colors and Times text as in the paper)."""
    work = out_pdf.parent / ("_" + out_pdf.stem)
    work.mkdir(parents=True, exist_ok=True)
    picture = picture.replace(r"Section~\ref{sec:guard}", "Section~" + guard_section)
    doc = r"""\documentclass[10pt,border=1pt]{standalone}
\usepackage{iftex}
\ifXeTeX
  \usepackage[no-math]{fontspec}
  \setmainfont{texgyretermes}[Extension=.otf, UprightFont=*-regular, BoldFont=*-bold,
                              ItalicFont=*-italic, BoldItalicFont=*-bolditalic]
\else
  \usepackage[T1]{fontenc}\usepackage{newtxtext}
\fi
\usepackage{xcolor}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,fit,calc,shapes.geometric,backgrounds}
\definecolor{fgblue}{HTML}{2A78D6}
\definecolor{fgorange}{HTML}{EB6834}
\definecolor{fgink}{HTML}{0B0B0B}
\definecolor{fgmuted}{HTML}{898781}
\definecolor{fgfill}{HTML}{EEF4FC}
\definecolor{fggrey}{HTML}{F4F3F0}
\begin{document}
""" + picture + "\n\\end{document}\n"
    (work / "d.tex").write_text(doc, encoding="utf-8")
    r = subprocess.run([str(TECTONIC), "d.tex"], cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("TikZ render failed:\n" + r.stderr[-2000:])
    shutil.copy(work / "d.pdf", out_pdf)
    shutil.rmtree(work)


def main(build: Path, ref: Path, out: Path) -> None:
    abstract, body = access_body(build)
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)

    # section label needed inside the architecture diagram (IEEE Access numbering)
    sec = re.findall(r"\\section\{([^}]*)\}", body)
    subs_before_guard = body[:body.index(r"\label{sec:guard}")]
    sys_idx = sec.index(next(s for s in sec if "System" in s))
    guard_letter = "ABCDEFGHIJ"[subs_before_guard[subs_before_guard.rindex(r"\section{"):].count(r"\subsection{") - 1]
    guard_section = f"{ROMAN[sys_idx]}-{guard_letter}"

    # 1) TikZ diagrams -> PDF
    pics = re.findall(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", body, re.S)
    assert len(pics) == len(TIKZ), len(pics)
    for pic, name in zip(pics, TIKZ):
        render_tikz(pic, out / name, guard_section)
        body = body.replace(pic, r"\includegraphics{" + name + "}")

    # 2) raster/vector figures, renamed in figure order; screenshots as JPEG (fast to embed)
    for old, new in FIGURES.items():
        assert body.count(old) == 1, old
        body = body.replace(old, new)
        src = HERE / old
        if new.endswith(".jpg") and src.suffix == ".png":
            img = cv2.imread(str(src))
            cv2.imwrite(str(out / new), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            shutil.copy(src, out / new)

    # 3) reference-style section banners, all floats placed before the conclusion
    k = 0

    def banner(m):
        nonlocal k
        title = m.group(1)
        k += 1
        return ("%=========================================================\n"
                f"% SECTION {ROMAN[k - 1]} - {re.sub(r'[^A-Za-z -]', '', title).upper()}\n"
                "%=========================================================\n" + m.group(0))

    body = re.sub(r"(?m)^\\section\{([^}]*)\}", banner, body)
    body = body.replace("%=========================================================\n% SECTION " + ROMAN[k - 1],
                        "\\FloatBarrier\n\n%=========================================================\n% SECTION " + ROMAN[k - 1], 1)
    body = body.replace(r"\begin{thebibliography}", DATA_AVAILABILITY + "%=========================================================\n"
                        "% REFERENCES\n%=========================================================\n\\begin{thebibliography}", 1)
    body = body.replace(r"\begin{keywords}", "%=========================================================\n"
                        "% INDEX TERMS\n%=========================================================\n\\begin{keywords}", 1)
    body = body.replace("% ===================================================================================================" "==\n", "")
    body = re.sub(r"(?m)^% =+\n", "", body)

    (out / "access.tex").write_text(PREAMBLE + body + ENDING, encoding="utf-8")

    # author photographs (paper/authors/, git-ignored) for the biographies
    for photo in sorted((HERE / "authors").glob("author_*.jpg")):
        if photo.stem in (ENDING + PREAMBLE):
            shutil.copy(photo, out / "figures" / photo.name)

    # 4) class, logos and icons from the reference project; logo names as the class requests them
    for f in ("ieeeaccess.cls", "IEEEtran.cls", "bullet.png", "orcid_id_image.jpg"):
        shutil.copy(ref / f, out / f)
    cls = (ref / "ieeeaccess.cls").read_text(encoding="utf-8", errors="replace")
    for want in set(re.findall(r"\\includegraphics\[[^]]*\]\{([A-Za-z]*[Ll]ogo\.png)\}", cls)):
        src = next((p for p in ref.iterdir() if p.name.lower() == want.lower()), None)
        if src is not None:  # (jtehmLogo.png belongs to another journal option and is not shipped)
            shutil.copy(src, out / want)

    zpath = out.with_suffix(".zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out).as_posix())
    print(f"[overleaf] {out}  ->  {zpath}  ({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
