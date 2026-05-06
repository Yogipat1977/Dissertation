import re
import os

def count_words_in_typ(filepath):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Strip comments
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    
    # Strip Typst syntax like #heading, #include, etc.
    # This is a bit rough but should get the main text 
    content = re.sub(r'#\w+(\(.*?\))?(\[.*?\])?', '', content) # Strip #func(...) or #func[...]
    content = re.sub(r'(\[|\])', '', content) # Strip brackets
    content = re.sub(r'\$', '', content) # Strip math
    
    words = content.split()
    return len(words)

files = [
    "chapters/01_introduction.typ",
    "chapters/02_lit_review.typ",
    "chapters/03_methodology.typ",
    "chapters/04_results.typ",
    "chapters/05_evaluation.typ",
    "chapters/06_conclusion.typ",
    "chapters/appendix-a.typ"
]

total = 0
print(f"{'File':<30} | {'Word Count':<10}")
print("-" * 45)
for f in files:
    count = count_words_in_typ(f)
    print(f"{f:<30} | {count:<10}")
    total += count

# Also count main.typ (Abstract and Acknowledgments)
main_count = count_words_in_typ("main.typ")
print(f"{'main.typ (front matter)':<30} | {main_count:<10}")
total += main_count

print("-" * 45)
print(f"{'Total':<30} | {total:<10}")
