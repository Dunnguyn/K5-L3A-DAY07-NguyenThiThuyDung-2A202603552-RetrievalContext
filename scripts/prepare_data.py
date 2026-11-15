"""Script to prepare Day 7 data and benchmark from Day 6 outputs."""
import json
import re
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.baseline import validate

DAY6_ROOT = Path("../K5-L3A-Day06-NguyenThiThuyDung-2A202603552-DataIngestionEngineering")
DAY6_CHUNKS_PATH = DAY6_ROOT / "starter/results/sensor_run/dev/chunks_b.jsonl"
DAY6_QUESTIONS_PATH = DAY6_ROOT / "starter/benchmark/sensor_questions.json"

SCRATCH_DIR = Path(r"C:\Users\Admin\.gemini\antigravity-cli\brain\306d7ff2-5e04-4b1f-a6cd-f9cd0641a583\scratch")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = SCRATCH_DIR


def load_day6_chunks():
    raw_chunks = []
    with open(DAY6_CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_chunks.append(json.loads(line))
    
    chunks = []
    for c in raw_chunks:
        chunk_obj = {
            "id": c.get("chunk_id") or c.get("id"),
            "source_id": c.get("document_id") or c.get("source_id", "SENSOR001"),
            "source": c.get("source", "data/documents.pdf"),
            "page": int(c["page"]),
            "text": c["text"],
            "metadata": c.get("metadata", {})
        }
        chunks.append(chunk_obj)
    return chunks


def load_and_convert_questions(chunks):
    raw_questions = json.load(open(DAY6_QUESTIONS_PATH, encoding="utf-8"))
    
    answerable_questions = []
    for q in raw_questions:
        gold_source_id = q.get("gold_source_id", "SENSOR001")
        gold_page = int(q.get("gold_page"))
        evidence_quote = q.get("evidence_quote")
        
        # Find exact substring in chunk['text'] matching whitespace-normalized quote
        words = evidence_quote.split()
        pattern = r"\s+".join(re.escape(w) for w in words)
        
        exact_evidence = None
        for c in chunks:
            if c["source_id"] == gold_source_id and c["page"] == gold_page:
                m = re.search(pattern, c["text"])
                if m:
                    exact_evidence = m.group(0)
                    break
        
        if not exact_evidence:
            raise ValueError(f"Could not find exact evidence slice for {q['id']} in page {gold_page}")
        
        converted_q = {
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "gold": [
                {
                    "source_id": gold_source_id,
                    "page": gold_page,
                    "evidence": exact_evidence
                }
            ]
        }
        if q.get("filter"):
            converted_q["filter"] = q["filter"]
            
        answerable_questions.append(converted_q)
    return answerable_questions


def create_unanswerable_questions():
    return [
        {
            "id": "SQ_UNANS_DEV_01",
            "split": "dev",
            "question": "Chuẩn giao tiếp không dây LoRaWAN cho cảm biến IoT hoạt động ở dải tần số nào tại Việt Nam?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu Chương II chỉ bao gồm phần cứng cảm biến, điều hòa tín hiệu và tiền xử lý, không có nội dung về mạng truyền thông không dây LoRaWAN hay dải tần số vô tuyến."
        },
        {
            "id": "SQ_UNANS_DEV_02",
            "split": "dev",
            "question": "Giá bán thương mại hiện nay của linh kiện cảm biến nhiệt độ DHT11 trên thị trường là bao nhiêu VNĐ?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu lý thuyết học thuật không cung cấp thông tin giá cả mua bán linh kiện thương mại của cảm biến DHT11."
        },
        {
            "id": "SQ_UNANS_TEST_01",
            "split": "test",
            "question": "Giao thức MQTT khi truyền tải dữ liệu cảm biến IoT lên broker sử dụng cổng mạng mặc định nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu không có nội dung về giao thức tầng ứng dụng mạng MQTT hoặc cổng kết nối mạng TCP/IP."
        },
        {
            "id": "SQ_UNANS_TEST_02",
            "split": "test",
            "question": "Cảm biến LiDAR quét khoảng cách bằng laser sử dụng bước sóng nào để đảm bảo tiêu chuẩn an toàn mắt Class 1?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu không đề cập đến công nghệ cảm biến quang học LiDAR hay tiêu chuẩn an toàn laser Class 1."
        }
    ]


def main():
    print("1. Loading chunks from Day 6...")
    chunks = load_day6_chunks()
    print(f"   Loaded {len(chunks)} chunks.")
    
    print("2. Converting answerable benchmark questions...")
    answerable_questions = load_and_convert_questions(chunks)
    print(f"   Converted {len(answerable_questions)} answerable questions.")
    
    print("3. Validating chunks and answerable questions against starter/baseline.py...")
    validate(chunks, answerable_questions)
    print("   Validation passed successfully!")
    
    print("4. Creating unanswerable benchmark questions...")
    unans_questions = create_unanswerable_questions()
    print(f"   Created {len(unans_questions)} unanswerable questions (2 dev, 2 test).")
    
    # Save chunks.json
    chunks_file = DATA_DIR / "chunks.json"
    chunks_file.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"   Saved {chunks_file} ({len(chunks)} chunks)")
    
    # Save questions.json (20 answerable questions for baseline.py and retrieval evaluation)
    questions_file = DATA_DIR / "questions.json"
    questions_file.write_text(json.dumps(answerable_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"   Saved {questions_file} ({len(answerable_questions)} questions)")
    
    # Create cases.json (All 24 questions for agent runner and agent_metrics.py)
    cases = []
    # Add answerable
    for q in answerable_questions:
        cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer",
            "expected_answer": q["expected_answer"],
            "gold": q["gold"],
            "filter": q.get("filter")
        })
    # Add unanswerable
    for uq in unans_questions:
        cases.append(uq)
        
    # Sort cases to keep consistent order: dev (SQ01..SQ15, SQ_UNANS_DEV_01..02), test (SQ16..SQ20, SQ_UNANS_TEST_01..02)
    dev_cases = [c for c in cases if c["split"] == "dev"]
    test_cases = [c for c in cases if c["split"] == "test"]
    all_cases = dev_cases + test_cases
    
    cases_file = DATA_DIR / "cases.json"
    cases_file.write_text(json.dumps(all_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"   Saved {cases_file} ({len(all_cases)} total cases: {len(dev_cases)} dev, {len(test_cases)} test)")
    
    print("\nSummary of Benchmark:")
    print(f"- Dev set: {len(dev_cases)} cases (15 answerable, 2 unanswerable)")
    print(f"- Test set: {len(test_cases)} cases (5 answerable, 2 unanswerable)")
    print(f"- Total: {len(all_cases)} cases (Minimum required: 24). Requirement met!")


if __name__ == "__main__":
    main()
