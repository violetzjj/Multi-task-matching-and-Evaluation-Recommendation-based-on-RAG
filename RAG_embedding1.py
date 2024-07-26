import os
from RAG.Embeddings import OpenAIEmbedding
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
import tqdm

apikey = os.environ['OPENAI_API_KEY'] = 'your openai apikey'
baseurl = os.environ['OPENAI_BASE_URL']="openai baseurl or forwarding url"

def sanitize_filename(filename):
    return ''.join([c for c in filename if c.isalnum() or c in " _-"]).strip()

def main():
    embeddings_model = OpenAIEmbedding()


    file_path = 'data2/test_ca.txt'


    reader = ReadFiles(file_path)
    lines = reader.read_file_content(file_path).splitlines()
    lines_number = len(lines)



    for index,line in enumerate(lines,start=1):
        if line.strip():  

            vector_store = VectorStore(document=[line])


            vector = embeddings_model.get_embedding(line)
            vector_store.vectors.append(vector)

            
            storage_path = f'storage_all1/{index}'
            if not os.path.exists(storage_path):
                os.makedirs(storage_path)
            vector_store.persist(path=storage_path)
            print(f"Saved vector for line: {index}")

if __name__ == "__main__":
    main()