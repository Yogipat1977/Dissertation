// =============================================================================
// Academic Paper Template for Typst
// Provides: conf() function for structured paper formatting
// =============================================================================

#let conf(
  title: none,
  authors: (),
  abstract: none,
  keywords: (),
  bibliography-file: none,
  doc,
) = {
  // --- Page & Document Setup ---
  set document(
    author: authors.map(a => a.name),
  )

  set page(
    paper: "a4",
    margin: (left: 2.5cm, right: 2.5cm, top: 2.5cm, bottom: 2.5cm),
    numbering: "1",
    number-align: center,
  )

  // --- Typography ---
  set text(font: "Linux Libertine", size: 10pt, lang: "en")
  set par(justify: true, leading: 0.65em, first-line-indent: 0em)
  set heading(numbering: "1.")

  // Heading styling
  show heading.where(level: 1): it => {
    v(1.2em)
    text(size: 14pt, weight: "bold")[#it]
    v(0.6em)
  }
  show heading.where(level: 2): it => {
    v(0.8em)
    text(size: 12pt, weight: "bold")[#it]
    v(0.4em)
  }
  show heading.where(level: 3): it => {
    v(0.6em)
    text(size: 10pt, weight: "bold", style: "italic")[#it]
    v(0.3em)
  }

  // --- Title Block ---
  align(center)[
    #v(0.5em)
    #text(size: 16pt, weight: "bold")[#title]
    #v(1.5em)
  ]

  // --- Authors ---
  if authors.len() > 0 {
    align(center)[
      #text(size: 11pt)[
        #for (i, author) in authors.enumerate() {
          if i > 0 { h(1.5em) }
          [*#author.name*#super(author.affiliation)]
        }
        \
        \
        // Affiliations (deduplicate)
        #let affiliations = authors.map(a => a.affiliation).dedup()
        #for aff in affiliations {
          [#super(aff)School of Architecture, Computing and Engineering\
          University of East London, London, United Kingdom\ ]
        }
        // Emails
        #for author in authors {
          if author.email != "" {
            text(size: 9pt)[#author.email]
            h(1.5em)
          }
        }
      ]
      #v(1.5em)
    ]
  }

  // --- Abstract ---
  if abstract != none {
    align(center)[#text(weight: "bold", size: 11pt)[Abstract]]
    v(0.5em)
    par(first-line-indent: 0em)[#abstract]
    v(0.5em)
  }

  // --- Keywords ---
  if keywords.len() > 0 {
    text(size: 9pt, weight: "bold")[Keywords: ]
    text(size: 9pt)[#keywords.join(", ")]
    v(1.5em)
  }

  // --- Document Body ---
  doc

  // --- Bibliography ---
  if bibliography-file != none {
    bibliography(bibliography-file, style: "ieee")
  }
}
