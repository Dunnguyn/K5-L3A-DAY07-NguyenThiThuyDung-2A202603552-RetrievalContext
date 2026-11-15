# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Nguồn của bộ SENSOR001

Chương II về xử lý dữ liệu cảm biến do giảng viên cung cấp. Học viên đã ghi nhận quyền công khai trong [hồ sơ Day 6](https://github.com/Dunnguyn/K5-L3A-Day06-NguyenThiThuyDung-2A202603552-DataIngestionEngineering/blob/main/STUDENT.md). Đường dẫn PDF trong trace tính từ gốc repo Day 6. Để chạy Day 7 chỉ cần ba tệp chunks.json, questions.json và cases.json trong thư mục này.

Khi có repo Day 6 nằm cạnh repo này, chạy `python scripts/prepare_data.py --output starter/results/day6-export` để xuất lại từ chunks_b.jsonl và sensor_questions.json. Bản benchmark đã chạy có hai sai khác so với bản xuất gốc. Ở trang 42, đoạn kết thúc bằng ID 0_393 thừa chữ “Kh” trước “khả năng sai sót cao”. Đoạn 393_547 cùng trang được gán nhầm trang 39. Giữ bản benchmark trong Git để kiểm chứng trace cũ. Bản xuất lại dùng nội dung và số trang từ Day 6.
