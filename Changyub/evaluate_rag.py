"""동일한 10개 질문을 RAG 유무로 비교하고 응답·근거·비용을 저장한다."""
import csv
import json
from datetime import datetime
from zoneinfo import ZoneInfo

from rag_pipeline import RAG, RESULTS, ABSTENTION

QUESTIONS = [
    dict(id="Q1", question="공식 튜토리얼에서 range(5)는 어떤 숫자를 만들며 끝값 5가 포함되나요?",
         doc="controlflow", heading="range", facts="0, 1, 2, 3, 4를 만들며 끝값 5는 제외한다."),
    dict(id="Q2", question="list.sort()와 sorted()는 원래 리스트 변경과 반환값이 어떻게 다른가요?",
         doc="datastructures", heading="리스트", facts="list.sort는 원본을 변경하고 None을 반환한다. sorted는 새 정렬 리스트를 반환한다."),
    dict(id="Q3", question="튜토리얼의 원소 한 개짜리 튜플 예제에서 쉼표는 왜 필요한가요?",
         doc="datastructures", heading="튜플", facts="괄호만으로는 튜플이 되지 않으므로 값 뒤에 쉼표를 붙인다."),
    dict(id="Q4", question="함수의 기본 인자에 빈 리스트를 쓰면 왜 호출 사이에 값이 누적되고 어떻게 고치나요?",
         doc="controlflow", heading="기본 인자", facts="기본값은 함수 정의 때 한 번 평가되어 공유된다. None을 기본값으로 두고 함수 안에서 새 리스트를 만든다."),
    dict(id="Q5", question="모듈 파일의 __name__은 직접 실행할 때와 import할 때 어떻게 다른가요?",
         doc="modules", heading="모듈을 스크립트", facts="직접 실행하면 __main__, import하면 모듈 이름이다."),
    dict(id="Q6", question="try문의 finally 절은 예외나 return이 있어도 언제 실행되나요?",
         doc="errors", heading="뒷정리", facts="예외 발생 여부와 관계없이 실행된다. return으로 떠나기 전에 실행된다."),
    dict(id="Q7", question="튜토리얼에서 17 // 3과 17 % 3의 결과는 각각 무엇인가요?",
         doc="introduction", heading="숫자", facts="몫은 5, 나머지는 2이다."),
    dict(id="Q8", question="함수 인자에서 / 앞과 * 뒤의 매개변수는 각각 어떻게 전달해야 하나요?",
         doc="controlflow", heading="특수 매개 변수", facts="/ 앞은 위치 전용, * 뒤는 키워드 전용이다."),
    dict(id="Q9", question="이 문서에서 설명하는 Django 모델의 select_related와 prefetch_related 차이는 무엇인가요?",
         doc=None, heading=None, facts="수집한 Python 튜토리얼 5개에 Django ORM 설명이 없어 답변을 유보한다."),
    dict(id="Q10", question="이 문서에 적힌 HateSlop 5기 담당 강사의 이름은 무엇인가요?",
         doc=None, heading=None, facts="수집 문서에 수업 담당 강사 정보가 없어 답변을 유보한다."),
]


def evaluate():
    rag = RAG()
    build = rag.build()
    answers, rows = [], []
    for q in QUESTIONS:
        for with_rag in [False, True]:
            result = rag.answer(q["question"], with_rag=with_rag)
            result.update(id=q["id"], expected_doc=q["doc"], expected_heading=q["heading"], expected_facts=q["facts"])
            answers.append(result)
            hits = [c for c in result["retrieved"] if c["doc_id"] == q["doc"]]
            rows.append(dict(id=q["id"], mode=result["mode"], in_corpus=q["doc"] is not None,
                             expected_doc_hit=bool(hits) if q["doc"] else "",
                             expected_section_hit=any(q["heading"] in c["heading"] for c in hits) if q["doc"] else "",
                             citation_count=len(result["sources"]), invalid_citations=len(result["invalid_citations"]),
                             abstained=ABSTENTION in result["answer"], tokens=result["tokens"],
                             latency_s=result["latency_s"], cached=result["cached"], answer=result["answer"],
                             source_urls=" | ".join(s["url"] for s in result["sources"])))
            print(f'{q["id"]} {result["mode"]}: {result["answer"][:100]}', flush=True)
    (RESULTS / "rag_answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS / "rag_comparison.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    supported = [r for r in rows if r["mode"] == "rag" and r["in_corpus"]]
    unsupported = [r for r in rows if r["mode"] == "rag" and not r["in_corpus"]]
    summary = dict(executed_at=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds"),
                   chat_model=rag.chat_model, embedding_model=rag.embed_model, top_k=rag.top_k,
                   retrieval="dense + BM25 RRF + function comparison subqueries + parent section context",
                   build=build, questions=len(QUESTIONS), answer_count=len(answers),
                   retrieval_doc_hit_at_k=sum(r["expected_doc_hit"] for r in supported) / len(supported),
                   retrieval_section_hit_at_k=sum(r["expected_section_hit"] for r in supported) / len(supported),
                   supported_answers_with_citations=sum(r["citation_count"] > 0 for r in supported),
                   invalid_citation_count=sum(r["invalid_citations"] for r in rows),
                   unsupported_abstention_rate=sum(r["abstained"] for r in unsupported) / len(unsupported),
                   new_api_calls=len(rag.api_calls), new_api_tokens=sum(r["tokens"] for r in rag.api_calls),
                   caveat="출처 번호가 유효하다는 것은 주장의 의미적 근거를 보장하지 않는다. 응답 내용은 별도 검토한다.")
    (RESULTS / "rag_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path = RESULTS / "rag_api_calls.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        for call in rag.api_calls:
            f.write(json.dumps({"executed_at": summary["executed_at"], **call}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    evaluate()
