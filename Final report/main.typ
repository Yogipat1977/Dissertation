#import "template.typ": *

#show: project.with(
  title: "Final Year Project Report", // Replace with your title
  author: "Yogi Amitkumar Patel", //
  student_id: "2536809", //
  degree: "Data Science and Artificial Intelligence", //
  supervisor: "Maimoona Sarif", //
  date: datetime(year: 2026, month: 3, day: 16), //

  abstract: [
    This is where your abstract goes. Remember, it must be 500 words or less. It is NOT an introduction. It needs three elements: Purpose, Methodology, and Outcome.
  ],

  acknowledgments: [
    Here you can thank the people, including your supervisor, who have helped you with your project.
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

