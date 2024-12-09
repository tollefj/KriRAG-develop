import dotenv
import nltk
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv()

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
# https://huggingface.co/intfloat/multilingual-e5-base/discussions/24
pr_number = 24
model = SentenceTransformer(
    EMBEDDING_MODEL,
    revision=f"refs/pr/{pr_number}",
    backend="openvino",  # we optimize cpu-inference to reduce docker container image (w/ cuda drivers etc.)
)

model.save_pretrained("sbert")
model_home = "models"
nltk.download("punkt", download_dir=model_home)
nltk.download("punkt_tab", download_dir=model_home)
