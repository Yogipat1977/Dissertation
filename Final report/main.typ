#import "template.typ": *

#show: project.with(
  title: "Final Year Project Report", // Replace with your title
  author: "Hard Joshi", // 
  student_id: "2512658", // 
  degree: "Data Science and Artificial Intelligence", // 
  supervisor: "Arish Siddiqui", // 
  date: datetime(year: 2025, month: 10, day: 13), // 
  
  abstract: [
    This is where your abstract goes. Remember, it must be 500 words or less. It is NOT an introduction. It needs three elements: Purpose, Methodology, and Outcome. [cite: 2, 3]
  ],
  
  acknowledgments: [
    Here you can thank the people, including your supervisor, who have helped you with your project. [cite: 5]
  ]
)

// --- MAIN BODY ---

#include "chapters/01_introduction.typ"

#include "chapters/02_lit_review.typ"

#include "chapters/03_methodology.typ"

#include "chapters/04_results.typ"

#include "chapters/05_evaluation.typ"

#include "chapters/06_conclusion.typ"

// --- REFERENCES ---
#bibliography("works.bib", style: "harvard-cite-them-right")

// --- APPENDICES ---
#show: appendix

// #include "chapters/appendix-a.typ"

= Code
Use code blocks for snippets. Do not dump 1000s of lines here; use appendices for that. [cite: 20, 21]

// Code block styling
#figure(
  caption: [Java Example],
  kind: raw,
  supplement: [Listing],
)[
java
static public void main(String[] args) {
  try {
    UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
  } catch(Exception e) {
    e.printStackTrace();
  }
  new WelcomeApp();
}

] // 

= Evaluation Include an evaluation of both product and process. Be objective. Reflect on what worked and what did not. 

= Conclusion Reflect on key findings, limitations, and future opportunities. 

// --- BIBLIOGRAPHY --- // Create a file named 'works.bib' in the same folder for this to work 
// #bibliography("works.bib", style: "harvard-cite-them-right") 
//

// --- APPENDICES --- #show: appendix

= Initial Project Proposal // Content for Appendix A 

= Final Project Proposal // Content for Appendix B 

= Application for Approval of Research Activities // Content for Appendix C 

= Client Consent Form // Content for Appendix D
