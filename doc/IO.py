import gpt as g

data = g.corr_io.reader(filepath)

for tag,corr in data.tags.items():
    print(f"{tag} \t {corr}")
