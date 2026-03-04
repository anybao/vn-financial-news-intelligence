from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SummarizationModel(str, Enum):
    """Available summarization models."""
    vit5 = "vit5"
    seq2seq = "seq2seq"
    extractive = "extractive"


class ArticleRequest(BaseModel):
    """Input payload for NLP processing endpoints."""
    text: str = Field(
        ...,
        title="Article Text",
        description="The full text of the financial news article to process.",
        min_length=10,
        json_schema_extra={"example": "Hòa Phát vừa công bố kết quả kinh doanh quý với lợi nhuận sau thuế đạt 2940 tỷ đồng, tăng 30% so với cùng kỳ năm trước."}
    )
    model: Optional[SummarizationModel] = Field(
        default=SummarizationModel.vit5,
        title="Summarization Model",
        description=(
            "Choose which summarization model to use:\n\n"
            "- **vit5** (default) — Pre-trained VietAI/vit5-base-vietnews-summarization (T5). "
            "Best quality, ~2s per request.\n"
            "- **seq2seq** — Custom BiLSTM Encoder + Bahdanau Attention + LSTM Decoder + Beam Search. "
            "Trained on 500 samples (educational demo). Falls back to extractive on degenerate output.\n"
            "- **extractive** — Unsupervised keyword TF + position + length + data scoring. "
            "Fastest (<0.1s), selects top-3 sentences."
        ),
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "text": "Hòa Phát vừa công bố kết quả kinh doanh quý với lợi nhuận sau thuế đạt 2940 tỷ đồng, tăng 30% so với cùng kỳ năm trước. Doanh thu thuần đạt 21165 tỷ đồng, vượt 21% so với kế hoạch.",
                "model": "vit5"
            },
            {
                "text": "VN-Index tăng 15 điểm nhờ nhóm ngân hàng dẫn dắt. Khối ngoại mua ròng 500 tỷ đồng.",
                "model": "seq2seq"
            },
        ]
    }}


class SummaryResponse(BaseModel):
    """Summarization result."""
    summary: str = Field(
        ...,
        title="Generated Summary",
        description="The generated summary text.",
        json_schema_extra={"example": "Hòa Phát báo lãi 2.940 tỷ đồng trong quý, tăng 30% so với cùng kỳ."}
    )
    model: SummarizationModel = Field(
        default=SummarizationModel.vit5,
        title="Model Used",
        description="The summarization model that produced this result.",
    )
    
class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    
class NERResponse(BaseModel):
    stocks: List[str]

class EventPredictionResponse(BaseModel):
    summary: str
    sentiment: str
    confidence: float
    stocks: List[str]
    is_duplicate: bool = False
