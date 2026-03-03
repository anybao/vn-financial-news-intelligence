#!/usr/bin/env python3
"""
Fake Data Generator for Financial News Intelligence Pipeline.

Generates synthetic Vietnamese financial news articles and pushes them
through the full NLP pipeline (Summarization, Sentiment, NER) via the API.

Usage:
    cd /Users/quanghuy/Documents/MSE/NLP501/FinalProject-NLP
    source venv/bin/activate
    export PYTHONPATH=$(pwd)
    python scripts/generate_fake_data.py --count 10
"""

import argparse
import json
import logging
import os
import random
import uuid
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Templates ──────────────────────────────────────────────────────────────────

STOCK_TICKERS = ["FPT", "VCB", "VNM", "VIC", "HPG", "TCB"]

STOCK_NAMES = {
    "FPT": "Tập đoàn FPT",
    "VCB": "Vietcombank",
    "VNM": "Vinamilk",
    "VIC": "Vingroup",
    "HPG": "Hòa Phát",
    "TCB": "Techcombank",
}

SOURCES = [
    "https://fake.vnexpress.net/rss/kinh-doanh.rss",
    "https://fake.vneconomy.vn/rss.rss",
    "https://cafef.vn/trang-chu.rss",
]

POSITIVE_TEMPLATES = [
    {
        "title": "Cổ phiếu {ticker} tăng mạnh sau báo cáo lợi nhuận quý vượt kỳ vọng",
        "summary": (
            "{name} vừa công bố kết quả kinh doanh quý với lợi nhuận sau thuế đạt {profit} tỷ đồng, tăng {pct}% so với cùng kỳ năm trước. "
            "Doanh thu thuần đạt {revenue} tỷ đồng, vượt {export_pct}% so với kế hoạch đề ra từ đầu năm. "
            "Giá cổ phiếu {ticker} đã tăng {price_pct}% trong phiên giao dịch sáng nay, đạt mức cao nhất trong 6 tháng qua. "
            "Khối lượng giao dịch đạt {volume} triệu cổ phiếu, gấp 3 lần trung bình 20 phiên gần nhất, cho thấy sự quan tâm lớn từ nhà đầu tư. "
            "Biên lợi nhuận gộp cải thiện đáng kể nhờ tối ưu hóa chuỗi cung ứng và giảm chi phí nguyên vật liệu đầu vào. "
            "Các chuyên gia phân tích tại công ty chứng khoán SSI và VNDirect đánh giá triển vọng tăng trưởng của {name} rất tích cực trong các quý tới. "
            "Chiến lược mở rộng thị trường sang khu vực Đông Nam Á cùng với việc đầu tư vào chuyển đổi số đang mang lại hiệu quả rõ rệt. "
            "Đồng thời, công ty cũng dự kiến trình đại hội cổ đông phương án chia cổ tức bằng tiền mặt tỷ lệ {div}% cho năm tài chính vừa qua."
        ),
    },
    {
        "title": "{name} ký hợp đồng hợp tác chiến lược trị giá {deal} tỷ đồng",
        "summary": (
            "{name} (mã {ticker}) hôm nay đã ký kết thỏa thuận hợp tác chiến lược với đối tác quốc tế trị giá {deal} tỷ đồng. "
            "Thỏa thuận kéo dài 5 năm, bao gồm chuyển giao công nghệ, đào tạo nhân sự và phát triển sản phẩm chung cho thị trường khu vực. "
            "Dự kiến hợp đồng này sẽ giúp công ty mở rộng thị phần tại khu vực Đông Nam Á và tăng doanh thu xuất khẩu lên {export_pct}% trong năm tới. "
            "Ông Nguyễn Văn Minh, Tổng Giám đốc {name}, cho biết đây là bước đi quan trọng trong chiến lược toàn cầu hóa của tập đoàn. "
            "Cổ phiếu {ticker} phản ứng tích cực với thông tin này, tăng {price_pct}% ngay trong phiên chiều với thanh khoản cao đột biến. "
            "Các tổ chức tài chính nước ngoài cũng đánh giá cao thương vụ, Goldman Sachs nâng khuyến nghị lên Mua với giá mục tiêu {price} nghìn đồng. "
            "Năm ngoái, {name} đã ghi nhận doanh thu {revenue} tỷ đồng và lợi nhuận ròng {profit} tỷ đồng, tăng trưởng {pct}% năm thứ ba liên tiếp. "
            "Với thương vụ mới này, ban lãnh đạo kỳ vọng doanh thu năm 2027 sẽ đạt mốc đột phá và khẳng định vị thế dẫn đầu ngành."
        ),
    },
    {
        "title": "Nhà đầu tư nước ngoài mua ròng mạnh cổ phiếu {ticker}",
        "summary": (
            "Khối ngoại đã mua ròng {volume} triệu cổ phiếu {ticker} trong tuần qua, nâng tổng giá trị sở hữu lên {own_pct}% vốn điều lệ. "
            "Đây là tuần mua ròng thứ tư liên tiếp của nhà đầu tư nước ngoài đối với cổ phiếu {ticker}, với tổng giá trị lũy kế đạt hơn {deal} tỷ đồng. "
            "Theo các chuyên gia tại Dragon Capital, dòng vốn ngoại đổ vào {name} phản ánh niềm tin vào tiềm năng tăng trưởng dài hạn của doanh nghiệp. "
            "Cổ phiếu {ticker} hiện đang giao dịch quanh mức {price} nghìn đồng, với P/E forward khoảng 12 lần, thấp hơn trung bình ngành. "
            "{name} vừa công bố lợi nhuận quý đạt {profit} tỷ đồng, vượt {export_pct}% so với ước tính của các công ty chứng khoán. "
            "Ban lãnh đạo công ty cho biết sẽ tiếp tục duy trì chính sách cổ tức ổn định với tỷ lệ chi trả tối thiểu {div}% mệnh giá. "
            "Ngoài ra, {name} đang triển khai kế hoạch mở rộng công suất nhà máy tại Bình Dương và Long An, dự kiến hoàn thành trong quý 3 năm nay. "
            "Giới phân tích nhận định rằng xu hướng mua ròng của khối ngoại sẽ tiếp tục trong bối cảnh Việt Nam đang tiến gần hơn đến việc được nâng hạng thị trường."
        ),
    },
]

NEGATIVE_TEMPLATES = [
    {
        "title": "Cổ phiếu {ticker} giảm sàn sau thông tin điều tra từ cơ quan quản lý",
        "summary": (
            "Cổ phiếu {ticker} của {name} đã giảm kịch sàn trong phiên giao dịch hôm nay sau khi có thông tin cơ quan quản lý bắt đầu rà soát hoạt động tài chính của công ty. "
            "Khối lượng bán tháo đạt {volume} triệu cổ phiếu, gấp 5 lần trung bình 20 phiên, cho thấy tâm lý hoảng loạn lan rộng trong nhóm nhà đầu tư cá nhân. "
            "Ủy ban Chứng khoán Nhà nước cho biết đang phối hợp với các cơ quan liên quan để xác minh thông tin và sẽ công bố kết quả trong thời gian sớm nhất. "
            "Nhiều nhà đầu tư lo ngại về rủi ro pháp lý và khả năng ảnh hưởng đến kết quả kinh doanh các quý tới nếu cuộc điều tra kéo dài. "
            "Trước đó, {name} đã ghi nhận doanh thu {revenue} tỷ đồng trong quý gần nhất nhưng biên lợi nhuận ròng giảm {cost_pct}% so với kỳ trước. "
            "Công ty chứng khoán VNDirect đã hạ khuyến nghị đối với {ticker} từ Mua xuống Trung lập, với giá mục tiêu mới là {new_target} nghìn đồng. "
            "Ban lãnh đạo {name} đã ra thông cáo khẳng định hoạt động kinh doanh vẫn diễn ra bình thường và sẵn sàng hợp tác với cơ quan chức năng. "
            "Tuy nhiên, áp lực bán vẫn rất lớn và nhiều quỹ đầu tư trong nước đã giảm tỷ trọng nắm giữ cổ phiếu {ticker} trong danh mục."
        ),
    },
    {
        "title": "{name} báo lỗ {loss} tỷ đồng trong quý gần nhất",
        "summary": (
            "{name} (mã {ticker}) vừa công bố báo cáo tài chính quý với khoản lỗ ròng {loss} tỷ đồng, đánh dấu quý thua lỗ thứ hai liên tiếp. "
            "Doanh thu thuần chỉ đạt {revenue} tỷ đồng, giảm {pct}% so với cùng kỳ và thấp hơn nhiều so với kế hoạch ban đầu. "
            "Nguyên nhân chính đến từ chi phí nguyên vật liệu tăng {cost_pct}%, chi phí lãi vay tăng đáng kể và doanh thu sụt giảm do nhu cầu thị trường yếu. "
            "Tỷ lệ nợ xấu của {name} cũng tăng từ 1.2% lên 2.8%, buộc công ty phải trích lập dự phòng thêm hàng trăm tỷ đồng. "
            "Ban lãnh đạo cho biết đang triển khai kế hoạch cắt giảm chi phí hoạt động từ 15-20% và tái cơ cấu bộ máy nhân sự để phục hồi trong nửa cuối năm. "
            "Các chuyên gia phân tích tại SSI Research nhận định {name} cần ít nhất 2-3 quý để quay lại quỹ đạo tăng trưởng, với điều kiện thị trường ổn định. "
            "Cổ phiếu {ticker} đã giảm {drop_pct}% kể từ đầu năm, hiện giao dịch quanh mức {new_target} nghìn đồng, thấp nhất trong 18 tháng qua. "
            "Nhiều cổ đông lớn bày tỏ quan ngại và yêu cầu ban lãnh đạo trình bày kế hoạch phục hồi chi tiết tại đại hội cổ đông bất thường sắp tới."
        ),
    },
    {
        "title": "Áp lực bán gia tăng, {ticker} mất {drop_pct}% giá trị trong tuần",
        "summary": (
            "Cổ phiếu {ticker} tiếp tục chịu áp lực bán mạnh, giảm {drop_pct}% trong tuần giao dịch vừa qua xuống còn {new_target} nghìn đồng mỗi cổ phiếu. "
            "Khối lượng giao dịch bình quân đạt {volume} triệu cổ phiếu mỗi phiên, trong đó bên bán chiếm ưu thế tuyệt đối với tỷ lệ 70-30. "
            "{name} đang đối mặt với nhiều thách thức bao gồm cạnh tranh gia tăng từ các đối thủ trong nước và quốc tế, lãi suất vay cao kỷ lục và nhu cầu tiêu dùng suy giảm. "
            "Báo cáo tài chính gần nhất cho thấy doanh thu của {name} chỉ đạt {revenue} tỷ đồng, giảm {pct}% so với cùng kỳ năm ngoái. "
            "Biên lợi nhuận gộp thu hẹp đáng kể từ 35% xuống còn 22% do chi phí đầu vào tăng mạnh trong bối cảnh giá nguyên liệu thế giới biến động. "
            "Giới phân tích hạ dự báo giá mục tiêu từ {old_target} nghìn đồng xuống {new_target} nghìn đồng, đồng thời cảnh báo rủi ro giảm giá thêm trong ngắn hạn. "
            "Khối ngoại cũng bán ròng {volume} triệu cổ phiếu {ticker} trong 3 tuần liên tiếp, phản ánh tâm lý thận trọng của nhà đầu tư tổ chức quốc tế. "
            "Nhiều chuyên gia khuyến nghị nhà đầu tư nên chờ đợi tín hiệu phục hồi rõ ràng hơn trước khi cân nhắc mua vào cổ phiếu {ticker}."
        ),
    },
]

NEUTRAL_TEMPLATES = [
    {
        "title": "Thị trường chứng khoán Việt Nam biến động nhẹ, VN-Index quanh ngưỡng {vni} điểm",
        "summary": (
            "VN-Index kết thúc phiên giao dịch hôm nay tại mức {vni} điểm, tăng nhẹ 0.{small_pct}% so với phiên trước trong bối cảnh thị trường thiếu vắng thông tin hỗ trợ rõ ràng. "
            "Thanh khoản đạt {liquidity} tỷ đồng trên sàn HOSE, giảm nhẹ so với mức trung bình 20 phiên gần nhất cho thấy tâm lý chờ đợi của nhà đầu tư. "
            "Nhóm cổ phiếu ngân hàng và bất động sản phân hóa mạnh, trong đó VCB và TCB giữ được sắc xanh trong khi VIC và HPG giao dịch quanh tham chiếu. "
            "Nhóm công nghệ giữ được đà tích cực với FPT tiếp tục là trụ cột chính, đóng góp hơn 2 điểm cho VN-Index trong phiên sáng. "
            "Khối ngoại giao dịch khá cân bằng với giá trị mua ròng khoảng {volume} triệu cổ phiếu, tập trung chủ yếu ở nhóm vốn hóa lớn VN30. "
            "Tỷ giá USD/VND dao động quanh mức {fx} đồng, không tạo áp lực đáng kể lên dòng vốn ngoại trên thị trường chứng khoán. "
            "Nhà đầu tư đang chờ đợi dữ liệu kinh tế vĩ mô bao gồm CPI, PMI và tăng trưởng GDP quý sắp công bố để xác định xu hướng ngắn hạn. "
            "Giới phân tích nhận định VN-Index sẽ tiếp tục dao động trong biên độ hẹp {vni}-{vni} điểm cho đến khi có thông tin xúc tác mới từ chính sách tiền tệ."
        ),
    },
    {
        "title": "Ngân hàng Nhà nước giữ nguyên lãi suất điều hành",
        "summary": (
            "Ngân hàng Nhà nước Việt Nam quyết định giữ nguyên các mức lãi suất điều hành trong phiên họp chính sách tiền tệ hôm nay, đúng như kỳ vọng của đa số chuyên gia. "
            "Lãi suất tái cấp vốn duy trì ở mức {rate}%/năm, lãi suất cho vay qua đêm liên ngân hàng ổn định quanh {rate}%/năm. "
            "Quyết định này phù hợp với mục tiêu cân bằng giữa hỗ trợ tăng trưởng kinh tế và kiểm soát lạm phát trong bối cảnh giá hàng hóa thế giới còn nhiều biến động. "
            "Tỷ giá USD/VND dao động quanh mức {fx} đồng, giảm nhẹ so với đầu tháng nhờ nguồn cung ngoại tệ dồi dào từ kiều hối và FDI giải ngân. "
            "Tăng trưởng tín dụng toàn hệ thống đạt khoảng 5.2% so với đầu năm, thấp hơn mục tiêu 14-15% do nhu cầu vay vốn từ doanh nghiệp còn yếu. "
            "Các ngân hàng thương mại lớn như VCB, TCB, VPB tiếp tục hạ lãi suất cho vay đối với lĩnh vực sản xuất kinh doanh và tiêu dùng. "
            "Chuyên gia kinh tế TS. Nguyễn Trí Hiếu nhận định Ngân hàng Nhà nước có thể xem xét giảm thêm lãi suất trong quý tới nếu lạm phát tiếp tục được kiềm chế. "
            "Thị trường trái phiếu Chính phủ cũng phản ứng tích cực, lợi suất kỳ hạn 10 năm giảm 15 điểm cơ bản xuống mức thấp nhất kể từ đầu năm."
        ),
    },
    {
        "title": "{name} tổ chức đại hội cổ đông thường niên năm 2026",
        "summary": (
            "{name} (mã {ticker}) vừa tổ chức đại hội đồng cổ đông thường niên năm 2026 tại Trung tâm Hội nghị Quốc gia TP.HCM với sự tham dự của hơn 500 cổ đông. "
            "Đại hội đã thông qua kế hoạch doanh thu {revenue} tỷ đồng và lợi nhuận {profit} tỷ đồng cho năm tài chính mới, tăng lần lượt {pct}% và {export_pct}% so với thực hiện năm trước. "
            "Hội đồng quản trị trình kế hoạch chia cổ tức bằng tiền mặt tỷ lệ {div}% mệnh giá và phát hành thêm cổ phiếu thưởng tỷ lệ {bonus}% từ nguồn thặng dư vốn cổ phần. "
            "Ban lãnh đạo cũng báo cáo kết quả kinh doanh năm 2025 với doanh thu và lợi nhuận đều vượt kế hoạch đề ra, khẳng định chiến lược phát triển bền vững của công ty. "
            "Về kế hoạch đầu tư, {name} dự kiến chi {deal} tỷ đồng cho việc nâng cấp hạ tầng công nghệ và mở rộng mạng lưới kinh doanh trên toàn quốc. "
            "Đại hội cũng bầu bổ sung 2 thành viên Hội đồng quản trị nhiệm kỳ 2024-2029 và thông qua việc lựa chọn đơn vị kiểm toán cho năm tài chính 2026. "
            "Cổ phiếu {ticker} phản ứng tích cực sau đại hội, tăng nhẹ {price_pct}% trong phiên chiều với thanh khoản cải thiện đáng kể. "
            "Các chuyên gia nhận định {name} đang có nền tảng tài chính vững chắc và kế hoạch kinh doanh khả thi để đạt mục tiêu tăng trưởng trong năm 2026."
        ),
    },
]


def generate_article(index: int) -> dict:
    """Generate a single fake financial news article."""
    sentiment_type = random.choice(["positive", "negative", "neutral"])

    if sentiment_type == "positive":
        template = random.choice(POSITIVE_TEMPLATES)
    elif sentiment_type == "negative":
        template = random.choice(NEGATIVE_TEMPLATES)
    else:
        template = random.choice(NEUTRAL_TEMPLATES)

    ticker = random.choice(STOCK_TICKERS)
    name = STOCK_NAMES[ticker]

    # Random params for templates
    params = {
        "ticker": ticker,
        "name": name,
        "profit": random.randint(500, 15000),
        "pct": random.randint(10, 80),
        "price_pct": round(random.uniform(2.0, 7.0), 1),
        "deal": random.randint(1000, 50000),
        "export_pct": random.randint(15, 40),
        "volume": round(random.uniform(1.0, 20.0), 1),
        "own_pct": round(random.uniform(5.0, 30.0), 1),
        "price": random.randint(20, 150),
        "loss": random.randint(200, 5000),
        "cost_pct": random.randint(15, 60),
        "drop_pct": round(random.uniform(5.0, 20.0), 1),
        "old_target": random.randint(50, 120),
        "new_target": random.randint(30, 80),
        "vni": random.randint(1200, 1400),
        "small_pct": random.randint(1, 9),
        "liquidity": random.randint(10000, 30000),
        "rate": round(random.uniform(3.5, 5.0), 1),
        "fx": random.randint(24500, 25500),
        "revenue": random.randint(5000, 100000),
        "div": random.randint(5, 30),
        "bonus": random.randint(5, 20),
    }

    title = template["title"].format(**params)
    summary = template["summary"].format(**params)

    # Generate a unique link
    slug = f"fake-news-{uuid.uuid4().hex[:8]}"
    published_dt = datetime.utcnow() - timedelta(minutes=random.randint(0, 120))
    published_str = published_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")

    return {
        "title": title,
        "link": f"https://fake-news.example.com/{slug}",
        "published": published_str,
        "summary": summary,
        "source_url": random.choice(SOURCES),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate fake financial news and push through the NLP pipeline.")
    parser.add_argument("--count", type=int, default=10, help="Number of fake articles to generate (default: 10)")
    args = parser.parse_args()

    logger.info(f"Generating {args.count} fake financial news articles...")
    articles = [generate_article(i) for i in range(args.count)]

    # Save to timestamped log file
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "fake_data")
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"fake_data_{timestamp}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(articles)} articles to {log_path}")

    # Print preview
    for i, a in enumerate(articles):
        logger.info(f"  [{i+1}] {a['title']}")

    # Push through the ingestion pipeline (same as real RSS flow)
    from src.ingestion.scheduler import save_articles
    save_articles(articles)

    logger.info("Done! Check the dashboard at http://localhost:8000/api/v1/dashboard")


if __name__ == "__main__":
    main()
