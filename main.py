import os

from dotenv import load_dotenv
import faiss
from huggingface_hub import login
import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline

class DocumentDatabase:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForSeq2SeqLM.from_pretrained(
        #     model_name).to(self.device)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config.attn_config['atten_impl'] = 'triton'
        self.config.init_device = self.device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype='auto',
            trust_remote_code=True,
        )
        self.index = None
        self.documents = None

    def create_database(self, documents, nlist=100, nprobe=10):
        self.documents = documents
        document_embeddings = self._encode(documents)
        self.index = self._create_faiss_index(document_embeddings, nlist, nprobe)

    def save_database(self, index_file, documents_file):
        faiss.write_index(self.index, index_file)
        with open(documents_file, 'w') as f:
            for document in self.documents:
                f.write(document + '\n')

    def load_database(self, index_file, documents_file):
        self.index = faiss.read_index(index_file)
        with open(documents_file, 'r') as f:
            self.documents = [line.strip() for line in f]

    def add_documents(self, documents):
        document_embeddings = self._encode(documents)
        self.index.add(document_embeddings)
        self.documents.extend(documents)

    def update_database(self, old_documents, new_documents):
        old_embeddings = self._encode(old_documents)
        new_index = faiss.IndexFlatL2(old_embeddings.shape[1])
        for i, doc in enumerate(self.documents):
            if doc not in old_documents:
                new_index.add(np.array([self.index.reconstruct(i)]))
        self.index = new_index
        self.add_documents(new_documents)

    def search(self, query, k=1, threshold=1.0):
        query_embedding = self._encode([query])
        D, I = self.index.search(np.array(query_embedding).astype('float32'), k)
        return [self.documents[i] for i in I[0] if D[0][0] < threshold]

    def _encode(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)
        embeddings = outputs.logits.mean(dim=1).to(torch.float32).cpu().numpy()
        return embeddings

    def _create_faiss_index(self, embeddings, nlist, nprobe):
        dimension = embeddings.shape[1]
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        assert not index.is_trained
        # Move the index to GPU for training
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.train(embeddings)
        assert gpu_index.is_trained
        # Move the index back to CPU after training
        index = faiss.index_gpu_to_cpu(gpu_index)
        index.add(embeddings)
        index.nprobe = nprobe
        return index


class LlamaV2Interface:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def generate_answer(self, context, query):
        inputs = self.tokenizer.encode(context + self.tokenizer.eos_token + query, return_tensors='pt').to(self.device)
        outputs = self.model.generate(inputs, max_length=1000, temperature=0.9)
        answer = self.tokenizer.decode(outputs[0])
        return answer

class lightblue_japanese_mpt_7b:
    def __init__(self, model_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.config.attn_config['atten_impl'] = 'triton'
        self.config.init_device = self.device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=self.config,
            torch_dtype='auto',
            trust_remote_code=True,
        )

    def generate_answer(self, context, query):
        pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)
        answer = pipe(context + self.tokenizer.eos_token + query, max_length=1000, temperature=0.9)
        return answer

# ファイルからテキストデータを抽出
def extract_text_from_files(file_paths):
    documents = []
    for file_path in file_paths:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
            documents.extend(df.values.flatten().tolist())
        elif file_path.endswith('.txt'):
            with open(file_path, 'r') as f:
                documents.append(f.read())
        elif file_path.endswith('.docx'):
            import docx2txt
            documents.append(docx2txt.process(file_path))
        else:
            raise ValueError(f'Unsupported file format: {file_path}')
    return documents

# .envファイルから環境変数を読み込む
load_dotenv()

# hugging_face_tokenを環境変数から読み込む
hugging_face_token = os.environ.get("hugging_face_token")
login(token=hugging_face_token)

# 使用例
file_paths = ["data/hyoujun_guideline_20210910.docx",]  # ファイルパスのリスト
documents = extract_text_from_files(file_paths)

# model = "meta-llama/Llama-2-7b-chat-hf"
model = "lightblue/japanese-mpt-7b"

db = DocumentDatabase(model)
db.create_database(documents, nlist=min(100, len(documents)), nprobe=10)
db.save_database('faiss_index', 'documents.txt')

# # データベースを読み込む
# db.load_database('faiss_index', 'documents.txt')

# # 新しいドキュメントを追加する
# new_documents = extract_text_from_files(["file4.xlsx", "file5.txt"])
# db.add_documents(new_documents)

# # データベースを更新する
# old_documents = ["old document 1", "old document 2", ...]
# new_documents = ["new document 1", "new document 2", ...]
# db.update_database(old_documents, new_documents)

# interface = LlamaV2Interface(model)
interface = lightblue_japanese_mpt_7b(model)

query = "デジタル・ガバメント推進標準ガイドラインとは何ですか？"
closest_documents = db.search(query, k=5, threshold=0.9)

if not closest_documents:
    print("Sorry, I couldn't find the information you're looking for.")
else:
    for doc in closest_documents:
        answer = interface.generate_answer(doc, query)
        print(answer)
