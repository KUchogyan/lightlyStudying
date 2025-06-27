from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 예제 텍스트
sentences = [
    ['i', 'love', 'machine', 'learning'],
    ['machine', 'learning', 'is', 'fun'],
    ['deep', 'learning', 'is', 'a', 'new', 'trend'],
    ['machine', 'learning', 'is', 'a', 'branch', 'of', 'artificial', 'intelligence']
]

# 모델 학습
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 단어 간의 유사도 계산
similarity = model.wv.similarity('machine', 'learning')
print(f'Similarity between "machine" and "learning": {similarity}')


# 단어 임베딩 시각화
def visualize_embeddingsz(embeddings, words):
    tsne = TSNE(n_components=2, perplexity=len(embeddings) - 1)
    embeddings_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(words):
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1])
        plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.show()


# 단어 리스트
words = list(model.wv.key_to_index)

# 단어 임베딩
embeddings = model.wv[words]

# 시각화
visualize_embeddingsz(embeddings, words)
