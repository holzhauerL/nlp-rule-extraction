"""
Convert .docx to .txt, based on this post: https://djangocentral.com/convert-a-docx-file-to-text-file/
"""


import docx2txt

# replace following line with location of your .docx file
MY_TEXT = docx2txt.process("EB115.docx")
with open("EB115.txt", "w") as text_file:
    print(MY_TEXT, file=text_file)