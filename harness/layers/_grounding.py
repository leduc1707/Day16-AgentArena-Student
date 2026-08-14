"""Cách bộ chấm đọc một trích dẫn — viết đúng một lần, hai lớp dùng chung.

`critic` (§2) và `citation_checker` (§11) hỏi hai câu khác nhau nhưng dựa
trên CÙNG một phép so sánh: "câu này có phải là bản sao nguyên văn của MỘT
DÒNG trong tài liệu X không?". Nếu hai lớp cài phép so sánh đó hơi khác
nhau thì chúng sẽ bất đồng đúng ở những claim ranh giới, nên nó nằm ở đây.

Phép so sánh được sao lại theo `arena/scorer.py` (đóng băng, đọc được):

  * `_norm`  -> NFC + casefold + gộp khoảng trắng (canonicalize — chuẩn
    hoá text về một dạng duy nhất trước khi so sánh).
  * `_norm_lines` + `_supports` -> so khớp bị GIỚI HẠN TRONG MỘT DÒNG, và
    câu ngắn hơn `MIN_SUPPORT_CHARS` không bao giờ được tính là trích dẫn.

Cố tình KHÔNG `from arena.scorer import ...`: `arena/` đóng băng nhưng
những tên bắt đầu bằng gạch dưới là nội bộ của nó, và một harness phụ
thuộc vào bộ chấm sẽ chết ngay khi vòng tính điểm chạy scorer ở tiến trình
khác. Chép lại 15 dòng rẻ hơn nhiều.

KHÔNG có hàm nào ở đây SỬA text của claim. Chúng chỉ trả lời có/không và
trả về `doc_id` — đúng hai loại sửa đổi mà README §8.2 cho phép.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"\s+")

#: Dưới ngưỡng này bộ chấm không coi là trích dẫn (arena.scorer.MIN_SUPPORT_CHARS).
MIN_SUPPORT_CHARS = 12


def norm(text) -> str:
    """Dạng chuẩn hoá mà mọi phép so sánh chuỗi của bộ chấm chạy trên đó."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", text).casefold()).strip()


def norm_lines(text: str) -> tuple:
    """Mỗi DÒNG của tài liệu, đã chuẩn hoá, bỏ dòng trống.

    Tách theo dòng chính là nửa quan trọng: nó chặn một câu vắt qua tiêu
    đề, một dòng trống và nửa đoạn văn được coi là "trích nguyên văn".
    """
    return tuple(line for line in (norm(raw) for raw in text.splitlines()) if line)


def quotes_a_line(lines, normalised_claim: str) -> bool:
    """Câu này có nằm gọn trong MỘT dòng của tài liệu không?"""
    if len(normalised_claim) < MIN_SUPPORT_CHARS:
        return False
    return any(normalised_claim in line for line in lines)


# ---------------------------------------------------------------------------
# Bộ nhớ đệm theo từng lượt chạy. Thuần tốc độ: nó không đổi một kết quả nào,
# chỉ tránh chuẩn hoá lại 120 body tài liệu ở mỗi claim.
# ---------------------------------------------------------------------------

_CACHE_KEY = "_grounding_cache"


def _cache(ctx) -> dict:
    state = getattr(ctx, "state", None)
    if not isinstance(state, dict):
        return {}
    cache = state.get(_CACHE_KEY)
    # `observations` dài thêm sau mỗi lượt tool, nên bộ đệm phải hết hạn
    # theo đúng số quan sát đã thấy.
    seen = len(getattr(ctx, "observations", ()) or ())
    if not isinstance(cache, dict) or cache.get("seen") != seen:
        cache = {"seen": seen, "observed": None, "bodies": None}
        state[_CACHE_KEY] = cache
    return cache


def observed(ctx) -> str:
    """`ctx.observed_text` đã chuẩn hoá."""
    cache = _cache(ctx)
    if cache.get("observed") is None:
        cache["observed"] = norm(getattr(ctx, "observed_text", "") or "")
    return cache["observed"]


def _bodies(ctx) -> list:
    """`[(doc, các dòng đã chuẩn hoá, body đã chuẩn hoá)]` theo thứ tự corpus."""
    cache = _cache(ctx)
    if cache.get("bodies") is None:
        corpus = getattr(ctx, "corpus", None)
        docs = list(getattr(corpus, "docs", ()) or ()) if corpus is not None else []
        cache["bodies"] = [(doc, norm_lines(doc.body), norm(doc.body)) for doc in docs]
    return cache["bodies"]


def any_doc_quotes(ctx, normalised_claim: str) -> bool:
    """Có BẤT KỲ tài liệu nào trong kho chứa nguyên văn câu này trên một dòng?

    Đây đúng là phép thử `HALLUCINATED` của bộ chấm (quét cả kho, không
    quan tâm đã truy xuất hay chưa). Sai ở phép thử này là loại claim đắt
    nhất: mất precision VÀ mất trọn 15 điểm honesty.
    """
    if not normalised_claim:
        return False
    return any(quotes_a_line(lines, normalised_claim) for _, lines, _ in _bodies(ctx))


def source_doc_id(ctx, normalised_claim: str):
    """`doc_id` của tài liệu ĐÃ QUAN SÁT thật sự chứa câu này, hoặc None.

    Hai điều kiện, cả hai đều bắt buộc:

      * `body` của tài liệu nằm nguyên vẹn trong các quan sát — nghĩa là
        nó đã về từ một lần fetch SẠCH. Một snippet của search hay một bản
        bị cắt không tính; gắn claim vào tài liệu lượt chạy chưa đọc bị
        chấm `UNRETRIEVED`.
      * câu nằm gọn trong MỘT dòng của body đó.
    """
    if not normalised_claim:
        return None
    seen = observed(ctx)
    if not seen:
        return None
    for doc, lines, body in _bodies(ctx):
        if body and body in seen and quotes_a_line(lines, normalised_claim):
            return doc.doc_id
    return None


def claim_text(claim) -> str:
    """`claim["text"]` nếu nó là chuỗi, ngược lại chuỗi rỗng."""
    if not isinstance(claim, dict):
        return ""
    value = claim.get("text")
    return value if isinstance(value, str) else ""


def sync_citations(report: dict) -> None:
    """`report["citations"]` = doc_id của các claim còn lại, giữ thứ tự.

    `citations` chỉ để tham khảo (bộ chấm không tính điểm nó), nhưng một
    báo cáo trích dẫn tài liệu mà không claim nào còn trỏ tới thì tự mâu
    thuẫn — và người đọc báo cáo là một phần của bài lab này.
    """
    citations: list[str] = []
    for claim in report.get("claims") or ():
        if not isinstance(claim, dict):
            continue
        doc_id = claim.get("doc_id")
        if isinstance(doc_id, str) and doc_id and doc_id not in citations:
            citations.append(doc_id)
    report["citations"] = citations
