import os
import fitz  # PyMuPDF
import re

def extract_methodology_from_pdfs(directory, output_file):
    """
    Extracts the methodology section from PDF files in a given directory and saves them to a text file.

    Args:
        directory (str): The path to the directory containing the PDF files.
        output_file (str): The path to the output text file.
    """
    # Keywords to identify the start of the methodology section (case-insensitive)
    start_keywords = [
        r"METHODOLOGY",
        r"METHODS",
        r"MATERIALS AND METHODS",
        r"PROPOSED METHOD",
        r"METHOD AND MATERIAL",
        r"RESEARCH METHODOLOGY",
        r"PROPOSED SYSTEM METHODOLOGY",
        r"PROPOSED METHODOLOGY",
        r"PROPOSED RESEARCH SECTION",
        r"IMPLEMENTATION & NETWORK ARCHITECTURE",
        r"PROPOSED APPROACH",
    ]

    # Keywords to identify the end of the methodology section (case-insensitive)
    end_keywords = [
        r"RESULTS",
        r"DISCUSSION",
        r"CONCLUSION",
        r"EXPERIMENTS",
        r"RESULTS AND DISCUSSION",
        r"EXPERIMENTAL RESULTS",
        r"EVALUATION",
    ]

    # Regex patterns for start and end keywords (ensuring they are section headers)
    # Regex patterns for start and end keywords (ensuring they are section headers, less restrictive)
    # This pattern looks for:
    # ^\s*: start of a line with optional whitespace
    # (?:(\d+\.?\s*)|(?:[IVXLCDM]+\.?\s*))?: optional section numbering (Arabic or Roman)
    # (?:" + "|".join(start_keywords) + r")(?:\s*|\Z)": any of the start keywords, followed by optional whitespace or end of string
    start_pattern = re.compile(r"^\s*(?:(\d+\.?\s*)|(?:[IVXLCDM]+\.?\s*))?(?:" + "|".join(start_keywords) + r")", re.IGNORECASE | re.MULTILINE)
    end_pattern = re.compile(r"^\s*(?:(\d+\.?\s*)|(?:[IVXLCDM]+\.?\s*))?(?:" + "|".join(end_keywords) + r")", re.IGNORECASE | re.MULTILINE)

    with open(output_file, "w", encoding="utf-8") as out_f:
        # Sort files for consistent order
        pdf_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(".pdf")])

        for filename in pdf_files:
            pdf_path = os.path.join(directory, filename)
            try:
                doc = fitz.open(pdf_path)
                out_f.write(f"--- START OF METHODOLOGY FROM: {filename} ---\n\n")
                
                full_text = ""
                for page in doc:
                    full_text += page.get_text("text") + "\n"

                # Find the start of the methodology section
                start_match = start_pattern.search(full_text)
                
                if not start_match:
                    out_f.write("Methodology section not found.\n\n")
                    continue

                # Text after the start of methodology
                text_after_start = full_text[start_match.end():]

                # Find the end of the methodology section
                end_match = end_pattern.search(text_after_start)

                if end_match:
                    # Extract text between the start and end keywords
                    methodology_text = text_after_start[:end_match.start()]
                else:
                    # If no end keyword is found, take a fixed number of characters (e.g., 5000)
                    # or until the end of the document as a fallback.
                    methodology_text = text_after_start

                out_f.write(methodology_text.strip())
                out_f.write(f"\n\n--- END OF METHODOLOGY FROM: {filename} ---\n\n")

            except Exception as e:
                out_f.write(f"Could not process {filename}. Error: {e}\n\n")
            finally:
                if 'doc' in locals() and doc:
                    doc.close()

if __name__ == "__main__":
    # The script is located in the same directory as the papers
    current_directory = os.path.dirname(os.path.abspath(__file__))
    output_txt_file = os.path.join(current_directory, "extracted_methods.txt")
    
    extract_methodology_from_pdfs(current_directory, output_txt_file)
    
    print(f"Extraction complete. The methodologies have been saved to '{output_txt_file}'")
