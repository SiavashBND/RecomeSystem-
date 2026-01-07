import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentRecommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df["overview"] = self.df["overview"].fillna("")
        self.df["genres"] = self.df["genres"].fillna("")
        self.df["soup"] = (self.df["genres"].astype(str) + " " + self.df["overview"].astype(str)).str.lower()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.tfidf = self.vectorizer.fit_transform(self.df["soup"])
        self.sim = cosine_similarity(self.tfidf)

        
        self.title_to_idx = {t: i for i, t in enumerate(self.df["title"])}

    def recommend(self, title: str, top_k: int = 10):
        if title not in self.title_to_idx:
            raise ValueError(f"Title not found: {title}")

        idx = self.title_to_idx[title]
        scores = list(enumerate(self.sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        scores = scores[1: top_k + 1]

        recs = []
        for i, s in scores:
            recs.append({
                "title": self.df.iloc[i]["title"],
                "similarity": float(s),
                "genres": self.df.iloc[i].get("genres", "")
            })
        return pd.DataFrame(recs)

if __name__ == "__main__":
    df = pd.read_csv("data/movies.csv")
    rec = ContentRecommender(df)

    print(rec.recommend("The Dark Knight", top_k=10))
