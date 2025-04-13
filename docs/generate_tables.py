import os

# Get current directory
folder_path = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(folder_path, "tables")
output_file = os.path.join(folder_path, "00_AllTables.tex")

def generate_latex_document(folder_path, output_file):
    tex_files = [f for f in os.listdir(folder_path) if f.endswith(".tex")]
    tex_files.sort()
    
    with open(output_file, "w") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage{lscape}\n")  # To handle large tables
        f.write("\\begin{document}\n")
        
        f.write("\\tableofcontents\n\\newpage\n")
        
        for tex_file in tex_files:
            if tex_file == "00_AllTables.tex":
                continue
            section_name = tex_file.replace(".tex", "").replace("_", " ")
            f.write(f"\\section{{{section_name}}}\n")
            f.write("\\input{" + os.path.join(folder_path, tex_file) + "}\n")
            f.write("\\newpage\n")
        
        f.write("\\end{document}\n")


if __name__ == "__main__":
    generate_latex_document(folder_path, output_file)
