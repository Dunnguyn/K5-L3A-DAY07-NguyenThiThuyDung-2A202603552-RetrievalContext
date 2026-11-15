# Day 7 · Agentic RAG trên bài giảng SENSOR001

**Nguyễn Thị Thùy Dung · 2A202603552**

## Dữ liệu và cách đo

Corpus kế thừa từ Day 6 gồm 43 đoạn trên 31 trang có nội dung được lập chỉ mục (trang 5–48), thuộc Chương II về xử lý dữ liệu cảm biến. Dữ liệu dùng cho bài tập lớp. Học viên đã ghi nhận quyền công khai trong hồ sơ Day 6. [Nguồn và sai khác khi chuyển dữ liệu](../../data/README.md). Benchmark gồm 15 câu có đáp án và 2 câu thiếu bằng chứng trên dev, 5 và 2 trên test. Bốn câu thiếu bằng chứng chỉ dùng để đánh giá agent.

Tất cả cấu hình dùng cùng câu hỏi gốc, không lọc metadata. Một hit cần khớp nguồn, trang và chứa nguyên văn một evidence hợp lệ. Context giữ nguyên từng đoạn, kèm ID, nguồn và trang theo thứ tự xếp hạng, bỏ qua đoạn vượt ngân sách. Thời gian là trung vị cho một query gồm tìm kiếm và đóng gói context, không gồm khởi tạo index. Chi phí API retrieval bằng 0 USD, chưa tính điện và phần cứng.

Dense dùng sentence-transformers 6.3.0 với paraphrase-multilingual-MiniLM-L12-v2 (384 chiều, revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42). ChromaDB 1.5.9 tạo collection trong bộ nhớ từ 43 đoạn, so khoảng cách cosine sau khi chuẩn hóa vector. BM25 dùng Unicode NFC, chuyển chữ thường, giữ dấu tiếng Việt, k1=1,5 và b=0,75. RRF cộng 1/(60+hạng) trên top 10 mỗi nhánh BM25 và dense, rồi lấy top 3. TF-IDF và các nhánh độc lập cũng trả top 3.

## Kết quả retrieval

| Split | Cấu hình | Budget ký tự | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Trung vị (ms) |
|---|---|---:|---:|---:|---:|---:|
| Dev | TF-IDF | 800 | 13/15 (87%) | 14/15 (93%) | 13/15 (87%) | 0,38 |
| Dev | BM25 | 800 | 13/15 (87%) | 13/15 (87%) | 13/15 (87%) | 0,42 |
| Dev | Dense | 800 | 10/15 (67%) | 12/15 (80%) | 11/15 (73%) | 48,10 |
| Dev | RRF | 800 | 11/15 (73%) | 14/15 (93%) | 11/15 (73%) | 55,08 |
| Dev | RRF | 3.200 | 11/15 (73%) | 14/15 (93%) | 14/15 (93%) | 58,04 |
| Dev | BM25 | 3.200 | 13/15 (87%) | 13/15 (87%) | 13/15 (87%) | 0,42 |
| Test | TF-IDF | 800 | 3/5 (60%) | 4/5 (80%) | 4/5 (80%) | 0,28 |
| Test | RRF | 3.200 | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 36,67 |

Tăng budget là phép thử giữ nguyên ranking cho từng phương pháp. Với RRF, ba câu có evidence sẵn trong top 3 được đưa thêm vào context. Với BM25, tăng budget không cứu được hai câu còn thiếu vì evidence nằm ngoài top 3. Chọn RRF 3.200 trên dev theo thứ tự ưu tiên Context Evidence Rate, Hit@3, Hit@1 rồi độ trễ. Quyết định được lưu trước lần chạy test trong phiên này. Workspace cũ đã có thư mục kết quả baseline test. Tập test chỉ có năm câu có đáp án, nên một câu làm tỷ lệ thay đổi 20 điểm phần trăm.

### Hai lỗi dev

- **SQ04, quan hệ do hàm truyền mô tả.** RRF xếp trang 5 và 34 trước trang 20 chứa định nghĩa. Với 800 ký tự, chỉ đoạn trang 5 vào context. Budget 3.200 giữ thêm trang 20, nên câu này chuyển từ thiếu sang đủ evidence dù Hit@1 vẫn trượt.
- **SQ05, hai đại lượng của đường cong chuẩn.** Trang 21 đứng thứ hai. Sau đoạn trang 20, phần ngân sách còn lại không chứa được toàn bộ đoạn trang 21 và nhãn nguồn. Bộ đóng gói bỏ qua trang 21 rồi lấy đoạn ngắn hơn ở trang 14. Khi tăng budget, trang 21 được giữ. Giới hạn còn lại là lỗi ranking như SQ02, nơi cả ba đoạn đầu đều trượt.

### Đọc context của từng câu test

Thang 0/1/2 tương ứng thiếu, có một phần và đủ bằng chứng kèm nguồn. Nhãn dưới đây được đối chiếu trực tiếp với văn bản context của hai cấu hình.

| Câu | TF-IDF | RRF | Căn cứ chấm |
|---|---:|---:|---|
| SQ16 | 2 | 2 | Trang 34 nêu cả ổn định và chính xác khi lọc nhiễu analog. |
| SQ17 | 1 | 2 | TF-IDF lấy trang 39 nói chung hai chiều ADC/DAC. RRF lấy thêm trang 36 chỉ rõ ADC đổi tương tự sang số. |
| SQ18 | 2 | 2 | Trang 37 liệt kê đủ lấy mẫu, lượng tử hóa và mã hóa theo thứ tự. |
| SQ19 | 2 | 2 | Trang 45 phân biệt ba loại dữ liệu cần loại bỏ gồm không chính xác, không cần thiết và lặp lại. |
| SQ20 | 2 | 2 | Trang 47 nêu cùng thang đo và mục đích dễ phân tích. RRF giữ trang này ở hạng ba. |

TF-IDF có 0/1/4 câu ở mức 0/1/2. RRF có 0/0/5. TF-IDF trả kết quả nhanh hơn, còn RRF với budget lớn phục hồi được định nghĩa trực tiếp ở SQ17.

## Agent và giới hạn đánh giá

Controller Gemini dùng function calling, buộc lượt đầu gọi search, sau đó cho phép tìm tiếp hoặc trả lời. Tối đa bốn lượt tìm. Mỗi lượt nhận context tích lũy trong budget. Agent đối chiếu ID citation với context và câu trả lời, rồi từ chối khi thiếu citation hợp lệ hoặc vượt giới hạn. Việc kiểm tra ngữ nghĩa của citation cần đọc câu trả lời và nguồn.

Gemini Flash Lite (gemini-flash-lite-latest) đã hoàn thành 48 lượt đánh giá. Google trả 404 cho Gemini 2.5 Flash với khóa này. Lỗi 429/503 tạm thời đã được chạy lại trên cùng cấu hình. Bảng ghi lượt thành công cuối cùng theo từng câu, gồm thời gian chờ trong lượt đó. Token theo usageMetadata của các lượt được giữ lại. Chi phí thanh toán thực tế và token của các lần chạy lỗi trước đó chưa xác định.

| Split · cấu hình | Có gọi search | Grounding số câu 0/1/2 | TB /2 | Citation hợp lệ | Câu trả lời có citation | Từ chối đúng | Từ chối nhầm | Trung vị giây | Tổng token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dev · tfidf | 17/17 | 8/0/7 | 0.93 | 7/7 | 7/7 | 2/2 | 8/15 | 9.82 | 32290 |
| dev · rrf-wide | 17/17 | 3/0/12 | 1.60 | 12/12 | 12/12 | 2/2 | 3/15 | 3.63 | 42314 |
| test · tfidf | 7/7 | 4/0/1 | 0.40 | 1/1 | 1/1 | 2/2 | 4/5 | 12.46 | 15437 |
| test · rrf-wide | 7/7 | 2/0/3 | 1.20 | 3/3 | 3/3 | 2/2 | 2/5 | 2.57 | 20629 |

Citation tính theo danh sách ID nguồn của từng câu trả lời, gộp các lần nhắc lại cùng ID. Đối chiếu câu trả lời với đoạn nguồn và đáp án cho thấy RRF trả lời đúng thêm SQ16 (trang 34) và SQ19 (trang 45) trên test. Với SQ17, RRF từ chối ở bước kiểm tra citation. Với SQ20, agent hết bốn lượt tìm. Hai ca này cho thấy đạt Context Evidence Rate 5/5 với query gốc chưa bảo đảm trả lời đúng khi model tự viết query.

Mỗi tệp agent-*.json trong thư mục kết quả chứa câu hỏi, truy vấn tool, kết quả, context và câu trả lời. Tệp reviews-*.json ghi điểm, nhãn từng citation và lý do theo ID câu. Tệp metrics-*.json tổng hợp chỉ số và phân bố số lượt tìm.

## Đối chiếu với vector store Day 6

Chạy lại truy vấn gốc trên bản sao collection **day06_sensor_b** của Day 6, kiểm tra model, 384 chiều, khoảng cách cosine, nguồn và 43 ID. Cả **50/50** danh sách top 3 trùng thứ tự với trace đã nộp (15 câu dev × 3 cấu hình và 5 câu test × 1 cấu hình). Phép đối chiếu chỉ kiểm tra thứ hạng. Các trace agent và thời gian ở trên thuộc lần chạy ban đầu. [Kết quả theo query](../results/day6-store-check.json)

## Tái lập

Từ thư mục gốc, dùng Python của môi trường Day 6 hoặc môi trường có sentence-transformers==6.3.0, chromadb==1.5.9 và python-dotenv nếu dùng tệp .env. Các lệnh dưới đây giả định đã kích hoạt môi trường. Trên Windows đặt PYTHONUTF8=1 trước khi chạy test. Khi đã có model trong cache có thể đặt HF_HUB_OFFLINE=1.

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python starter/experiment.py dev
python starter/experiment.py test
python starter/gemini_agent.py dev --model gemini-flash-lite-latest
python starter/gemini_agent.py test --model gemini-flash-lite-latest
```

Agent đọc GEMINI_API_KEY từ biến môi trường hoặc tệp .env cục bộ. Có thể thay tên model bằng model hỗ trợ function calling được cấp cho khóa. Sau khi có câu trả lời, ghi nhãn từng citation và từng câu theo BENCHMARK.md rồi tổng hợp bằng starter/agent_metrics.py. Trace retrieval gồm ranking, score, context, ID nguồn và thời gian từng query. [Kết quả và quyết định chọn cấu hình](../results/)


Trên PowerShell, đặt `$env:DAY7_RESULTS_DIR = "starter/results/reproduction"` trước chuỗi lệnh trên để thực sự chạy lại và lưu kết quả riêng. Agent dùng selection.json của lần chạy dev trong cùng thư mục và tiếp tục các câu còn thiếu khi chạy lại. Với bản sao Chroma Day 6, đặt `$env:DAY6_CHROMA_PATH` bằng đường dẫn thư mục chứa chroma.sqlite3. Để so sánh với trace đã nộp, dùng thư mục kết quả mặc định rồi chạy `python starter/check_day6_store.py`. Tên model Gemini là alias của nhà cung cấp nên nội dung lần chạy sau có thể khác. Phần nộp này thực hiện bài cơ bản.
