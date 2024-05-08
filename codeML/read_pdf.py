import tabula

file_path = "/home/adam/Documents/coding/genetics/results_Iceland.pdf"

df = tabula.read_pdf(file_path, pages=3)

df.to_csv("/home/adam/Documents/genetics/data_manipulation/percentage_SNP.csv")
