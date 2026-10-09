# FinEvent Financial Disclosure Annotation Guidelines

## 1. Mục đích & Phạm vi
Tài liệu này quy định tiêu chuẩn gán nhãn cho các đoạn trích từ hồ sơ SEC EDGAR (Form 8-K, Item 2.02, Item 5.02, Item 7.01, Exhibit 99.1) nhằm phục vụ nghiên cứu Financial NLP và phát hiện thay đổi dự báo tài chính (Guidance Revision).

---

## 2. Bản đồ Ontology & Phân loại Sự kiện

Mỗi đoạn trích phải được phân loại vào đúng 1 trong 5 lớp sau:

### 2.1. `GUIDANCE_ISSUANCE`
- **Định nghĩa**: Doanh nghiệp đưa ra dự báo mới cho một kỳ tài chính trong tương lai mà **không** đề cập đến việc điều chỉnh hoặc thay đổi một dự báo đã công bố trước đó.
- **Dấu hiệu**: "expects revenue between $X and $Y for Q2", "projects diluted EPS of $Z for FY2026".
- **Lưu ý**: Nếu chỉ có dự báo mới, gán `GUIDANCE_ISSUANCE`, KHÔNG gán `GUIDANCE_REVISION`.

### 2.2. `GUIDANCE_REVISION`
- **Định nghĩa**: Doanh nghiệp cập nhật, nâng (`raising`), hạ (`lowering`) hoặc tái khẳng định (`reaffirming`) một mức dự báo chính thức đã từng công bố cho **cùng kỳ tài chính mục tiêu**.
- **Dấu hiệu**: "revises guidance", "raises full-year revenue outlook to $X from previous $Y", "lowers gross margin expectation due to FX headwinds".
- **Bắt buộc**: Phải có bằng chứng về việc thay đổi so với kỳ vọng trước đó trong cùng ngữ cảnh.

### 2.3. `EARNINGS_RESULT`
- **Định nghĩa**: Báo cáo kết quả tài chính thực tế mà doanh nghiệp đã đạt được trong kỳ quá khứ vừa kết thúc.
- **Dấu hiệu**: "posted quarterly revenue of $X", "reported net income of $Y", "diluted EPS was $Z for the quarter ended September 30".
- **Nghiêm cấm**: Không đánh đồng Actual Results với Guidance.

### 2.4. `MANAGEMENT_CHANGE`
- **Định nghĩa**: Bổ nhiệm, từ nhiệm, nghỉ hưu hoặc thay đổi chức vụ của các lãnh đạo cấp cao (Hội đồng quản trị, CEO, CFO, COO, General Counsel).
- **Dấu hiệu**: "appointed as Chief Financial Officer", "announced the retirement of", "stepped down from the Board of Directors".

### 2.5. `OTHER`
- **Định nghĩa**: Các công bố doanh nghiệp trọng yếu khác không thuộc 4 nhóm trên (phát hành trái phiếu, mua lại cổ phiếu, cổ tức, kiện tụng, M&A).

---

## 3. Định danh Dự báo (Guidance Identity Tuple)
Mỗi sự kiện dự báo phải xác định đầy đủ bộ thuộc tính:
- `cik`: Mã định danh 10 chữ số của doanh nghiệp tại SEC.
- `metric`: Chỉ tiêu tài chính (`revenue`, `diluted_eps`, `gross_margin`, `operating_expenses`, `capex`, `dividend`, v.v.).
- `target_fiscal_period`: Kỳ tài chính mà con số hướng tới (ví dụ: `Q2 2026`, `FY 2026`).
- `accounting_basis`: `GAAP`, `NON_GAAP`, hoặc `UNKNOWN`.
- `currency`: `USD` hoặc đồng tiền liên quan.
- `unit`: `USD_B`, `USD_M`, `USD`, `PERCENT`, `PERCENTAGE_POINTS`, `BASIS_POINTS`, `USD_PER_SHARE`.
- `scope`: `COMPANY_WIDE` hoặc bộ phận cụ thể (`SEGMENT`).

---

## 4. Quy tắc Tính toàn vẹn Dẫn chứng (Evidence Integrity)
1. **Verbatim Match**: Chuỗi `evidence` phải trùng khớp 100% với lát cắt ký tự trên văn bản nguồn:
   ```python
   assert record["text"][record["char_start"]:record["char_end"]] == record["evidence"]
   ```
2. **Accession Traceability**: `accession_number` phải là số hồ sơ SEC thực tế (ví dụ: `0000320193-24-000106`).
3. **Trạng thái gán nhãn**:
   - `UNLABELED`: Dữ liệu thô vừa crawl.
   - `MACHINE_LABELED`: Nhãn do mô hình/regex đề xuất.
   - `PENDING_REVIEW`: Đã đưa vào hàng đợi kiểm duyệt.
   - `HUMAN_REVIEWED`: Đã qua chuyên viên kiểm tra và xác nhận thủ công.
   - `REJECTED`: Đoạn trích bị lỗi hoặc không đạt chuẩn.
