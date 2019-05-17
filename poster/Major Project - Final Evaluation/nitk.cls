
% LaTeX Template for National Institute of Technology
% Karnataka, Surathkal.
%
% Author: Adithya Bhat
% Email: <dth.bht@gmail.com>

\NeedsTeXFormat{LaTeX2e}

\LoadClass[12pt,a4paper]{article}
\ProvidesClass{nitk}

\RequirePackage{graphicx}
\RequirePackage[dvips]{geometry}
\RequirePackage{tikz}
\RequirePackage{setspace}

\usetikzlibrary{calc}

\newcommand*{\report}[1]{\gdef\@report{#1}%
}
\newcommand*{\rollno}[1]{\gdef\@rollno{#1}%
}
\newcommand*{\name}[1]{\gdef\@name{#1}%
}
\newcommand*{\guidea}[1]{\gdef\@guidea{#1}%
}
\newcommand*{\guideb}[1]{\gdef\@guideb{#1}%
}
\newcommand*{\dept}[1]{\gdef\@dept{#1}%
}
\newcommand*{\place}[1]{\gdef\@place{#1}%
}
\newcommand*{\years}[1]{\gdef\@years{#1}%
}

\def\and{\\ \vspace{0.5em}} % redefine so it is compatible with other classes

\setstretch{1.25}

\renewcommand{\maketitle}{% Make our beautiful front page
    \thispagestyle{empty}%
    \begin{tikzpicture}[remember picture, overlay]%
        \draw[line width = 1pt] ($(current page.north west) + (2em,-2em)$) rectangle ($(current page.south east) + (-2em,2em)$);%
    \end{tikzpicture}%
    
    {\centering
      \vspace{-2em}%
      {\Large\textbf{\@report}}\\%
      \vspace{1em}%
      On\\%
      \vspace{1em}%
      {\Huge \textbf{\@title}}\\%
      \vfill
      {\large \bfseries Submitted by}\\%
      \vspace{0.5em}%
      {\Large {\textbf{\@author}}}\\%
      \vspace{1em}%
     
      {\large \textit{Under the Guidance of}}\\%
      \vspace{1em}%
      {\Large \textbf{\@guidea}}\\%
      \vspace{1em}%
      {\Large \textbf{\@guideb}} \\%
      \vspace{1em}%
      {\large \textbf{\@dept, NITK, Surathkal}}\\%
      \vfill
      \vspace{1em}%
      {\large \textit{in partial fulfilment for the award of the degree}}\\%
      \textit{of}\\
      \vfill
      {\large \textbf{Bachelor of Technology}}\\
      in\\
      \vfill
      {\large \textbf{Information Technology}}\\
      at\\
      \vfill
      \begin{figure}[!h]%
          \centering
          \includegraphics{nitk-logo.png}%
      \end{figure}%
    %   \vspace{1em}%
    %   \vfill
    %   {\large \textbf{Date of Submission: \@date}}\\%
    %   \vspace{1em}%
    %   \vfill
    \vspace{1em}
      {\Large \bfseries Department of Information Technology}\\%
      \vspace{0.5em}%
      {\Large \bfseries National Institute of Technology Karnataka, Surathkal.}\\% Hard-coded NITK Name
      \vspace{0.5em}%
      {\large \bfseries \@years}%
      \newpage
      
    }
}

\let\origabstract\abstract
\let\origendabstract\endabstract


\renewcommand{\abstract}[1]{%
\pagenumbering{roman}
\section*{Abstract}
\addcontentsline{toc}{section}{Abstract}
#1
\newpage
}

\newcommand{\makedeclaration}[1]{%
% \pagenumbering{roman}
\addcontentsline{toc}{section}{Declaration}
\section*{Declaration}
#1\\

\begin{flushleft}
    Place: \@place\\
    Date: \@date
    \begin{flushright}
        ----------------------------- \\
        Signature of Student
    \end{flushright}
\end{flushleft}
\newpage
}

\newcommand{\acknowledge}[1]{
    % \thispagestyle{roman}
    \addcontentsline{toc}{section}{Acknowledgement}
    \section*{Acknowledgement}
    #1
    \newpage
}

\newcommand{\certificate}[1]{
    \addcontentsline{toc}{section}{Certificate}
    \section*{Certificate}
    #1
    \newpage
}

\let\oldtoc\tableofcontents
\renewcommand{\tableofcontents}{
    \thispagestyle{empty}
    \oldtoc
    \newpage
}

\let\oldlof\listoffigures
\renewcommand{\listoffigures}{
    \thispagestyle{empty}
    \oldlof
    \newpage
}

\let\oldlot\listoftables
\renewcommand{\listoftables}{
    \thispagestyle{empty}
    \oldlot
    \newpage
}

\newcommand{\references}[1]{
    \addcontentsline{toc}{section}{References}
    \begin{thebibliography}{1}
    #1
    \end{thebibliography}
    \newpage
}
