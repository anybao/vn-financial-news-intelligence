import torch
import re
from typing import List, Dict
from transformers import AutoModelForTokenClassification, AutoTokenizer
import logging

logger = logging.getLogger(__name__)

class HybridNERPredictor:
    """Extracts stock tickers using a hybrid rule-based dictionary and PhoBERT NER."""
    
    def __init__(self, model_path: str = "models/ner"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # VN30 stock dictionary with Vietnamese aliases
        self.stock_dictionary = {
            "ACB": ["ACB", "Ngân hàng Á Châu", "Asia Commercial Bank"],
            "BCM": ["BCM", "Becamex IDC", "Tổng Công ty Becamex IDC"],
            "BID": ["BID", "BIDV", "Ngân hàng Đầu tư và Phát triển Việt Nam"],
            "BVH": ["BVH", "Bảo Việt", "Tập đoàn Bảo Việt"],
            "CTG": ["CTG", "VietinBank", "Ngân hàng Công Thương Việt Nam"],
            "FPT": ["FPT", "Công ty Cổ phần FPT", "Tập đoàn FPT"],
            "GAS": ["GAS", "PV Gas", "Tổng Công ty Khí Việt Nam"],
            "GVR": ["GVR", "Cao su Việt Nam", "Tập đoàn Công nghiệp Cao su Việt Nam"],
            "HDB": ["HDB", "HDBank", "Ngân hàng Phát triển TP.HCM"],
            "HPG": ["HPG", "Hòa Phát", "Tập đoàn Hòa Phát"],
            "MBB": ["MBB", "MB Bank", "Ngân hàng Quân đội", "MB"],
            "MSN": ["MSN", "Masan", "Tập đoàn Masan"],
            "MWG": ["MWG", "Thế Giới Di Động", "Mobile World"],
            "PLX": ["PLX", "Petrolimex", "Tập đoàn Xăng dầu Việt Nam"],
            "POW": ["POW", "PV Power", "Tổng Công ty Điện lực Dầu khí Việt Nam"],
            "SAB": ["SAB", "Sabeco", "Tổng Công ty Bia Rượu Nước giải khát Sài Gòn"],
            "SHB": ["SHB", "Ngân hàng Sài Gòn - Hà Nội"],
            "SSB": ["SSB", "SeABank", "Ngân hàng Đông Nam Á"],
            "SSI": ["SSI", "Chứng khoán SSI", "Công ty Chứng khoán SSI"],
            "STB": ["STB", "Sacombank", "Ngân hàng Sài Gòn Thương Tín"],
            "TCB": ["TCB", "Techcombank", "Ngân hàng Kỹ thương Việt Nam"],
            "TPB": ["TPB", "TPBank", "Ngân hàng Tiên Phong"],
            "VCB": ["VCB", "Vietcombank", "Ngân hàng Ngoại thương Việt Nam"],
            "VHM": ["VHM", "Vinhomes", "Công ty Cổ phần Vinhomes"],
            "VIB": ["VIB", "Ngân hàng Quốc tế Việt Nam"],
            "VIC": ["VIC", "Vingroup", "Tập đoàn Vingroup"],
            "VJC": ["VJC", "Vietjet Air", "Vietjet", "Hãng hàng không Vietjet"],
            "VNM": ["VNM", "Vinamilk", "Công ty Cổ phần Sữa Việt Nam"],
            "VPB": ["VPB", "VPBank", "Ngân hàng Việt Nam Thịnh Vượng"],
            "VRE": ["VRE", "Vincom Retail", "Công ty Cổ phần Vincom Retail"]
        }
        
        try:
            # Initialize fine-tuned model (fallback to base model if not found for testing)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path).to(self.device)
            self.model.eval()
            self.has_model = True
        except Exception as e:
            logger.warning(f"Failed to load NER model from {model_path}: {e}. Running rule-based only.")
            self.has_model = False

    def predict_rule_based(self, text: str) -> List[str]:
        """Extract stocks based on exact string matching referencing the dictionary."""
        found_stocks = set()
        for ticker, aliases in self.stock_dictionary.items():
            for alias in aliases:
                # Add word boundary to regex to avoid partial matches
                if re.search(r'\b' + re.escape(alias) + r'\b', text, re.IGNORECASE):
                    found_stocks.add(ticker)
        return list(found_stocks)

    def predict_model_based(self, text: str) -> List[str]:
        """Extract ORG entities from text using fine-tuned Model and map them to stocks."""
        if not self.has_model:
            return []
            
        # Placeholder for TokenClassification pipeline logic
        # 1. Tokenize 2. Predict 3. Extract entities 4. Map back to Tickers
        return []

    def extract_stocks(self, text: str) -> Dict[str, List[str]]:
        """Combine rule-based and model-based predictions."""
        rule_based_stocks = self.predict_rule_based(text)
        model_based_stocks = self.predict_model_based(text)
        
        # Merge lists and deduplicate
        combined = list(set(rule_based_stocks + model_based_stocks))
        
        return {
            "stocks": combined
        }

if __name__ == "__main__":
    predictor = HybridNERPredictor()
    text = "Cổ phiếu VCB và FPT đều có dấu hiệu mua mạnh trong phiên sáng nay. Cổ phiếu Vinamilk thì đứng giá."
    result = predictor.extract_stocks(text)
    print(f"Extracted: {result}")
