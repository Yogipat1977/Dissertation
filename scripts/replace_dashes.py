import re

with open('Final report/chapters/appendix-a.typ', 'r') as f:
    content = f.read()

# Replace — with  - 
new_content = content.replace('—', ' - ')

with open('Final report/chapters/appendix-a.typ', 'w') as f:
    f.write(new_content)
