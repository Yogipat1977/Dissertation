#import "template.typ": *

#show: project.with(
  title: "Final Year Project Report", // Replace with your title
  author: "Yogi Amitkumar Patel", //
  student_id: "2536809", //
  degree: "Data Science and Artificial Intelligence", //
  supervisor: "Maimoona Sarif", //
  date: datetime(year: 2026, month: 3, day: 16), //

  abstract: [
  ],

  acknowledgments: [
  ],
)

// --- MAIN BODY ---

#include "chapters/01_introduction.typ"

#include "chapters/02_lit_review.typ"

#include "chapters/03_methodology.typ"

#include "chapters/04_results.typ"

#include "chapters/05_evaluation.typ"

#include "chapters/06_conclusion.typ"

#include "chapters/appendix-a.typ"
// --- REFERENCES ---
#bibliography("works.bib", style: "harvard-cite-them-right")

// --- APPENDICES ---
#show: appendix

