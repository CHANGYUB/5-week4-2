"""API를 사용하지 않는 청킹·본문 추출·영속 DB 회귀 검증."""
import tempfile
import unittest

import chromadb
from chromadb.config import Settings

from crawl import extract_sections
from rag_pipeline import chunk_text, hybrid_rank, parent_context_id


class ChunkTests(unittest.TestCase):
    def test_short_and_exact_text_have_no_duplicate_tail(self):
        self.assertEqual(chunk_text("", 5, 2), [])
        self.assertEqual(chunk_text("abcd", 5, 2), ["abcd"])
        self.assertEqual(chunk_text("abcde", 5, 2), ["abcde"])
        self.assertEqual(chunk_text("abcdefgh", 5, 2), ["abcde", "defgh"])

    def test_overlap_covers_original_text(self):
        text = "가나다라마바사아자차카타파하"
        pieces = chunk_text(text, 6, 2)
        restored = pieces[0] + "".join(piece[2:] for piece in pieces[1:])
        self.assertEqual(restored, text)

    def test_invalid_overlap_does_not_loop_forever(self):
        for size, overlap in [(0, 0), (4, 4), (4, -1), (4, 5)]:
            with self.assertRaises(ValueError):
                chunk_text("abc", size, overlap)


class CrawlTests(unittest.TestCase):
    def test_body_only_no_duplicate_blocks_and_preserve_code(self):
        html = '<nav>내비게이션</nav><div class="body"><section id="lists"><h2>리스트<a class="headerlink">¶</a></h2><ul><li><p>설명</p></li></ul><pre>if True:\n    print(1)</pre></section></div><footer>푸터</footer>'
        sections = extract_sections(html, "https://docs.python.org/test.html")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0]["url"], "https://docs.python.org/test.html#lists")
        self.assertEqual(sections[0]["text"].count("설명"), 1)
        self.assertIn("\n    print(1)", sections[0]["text"])
        self.assertNotIn("내비게이션", sections[0]["text"])
        self.assertNotIn("¶", sections[0]["heading"])


class HybridTests(unittest.TestCase):
    def test_child_section_gets_its_own_parent_not_same_number_in_other_doc(self):
        child = dict(id="c", doc_id="one", heading="4.9.3.3. 키워드", chunk=0, text="설명")
        candidates = [dict(id="wrong", doc_id="two", heading="4.9.3. 다른 절", chunk=0, text="가"*250),
                      dict(id="parent", doc_id="one", heading="4.9.3. 특수 매개 변수", chunk=0, text="가"*250)]
        self.assertEqual(parent_context_id(child, candidates), "parent")

    def test_distinct_sections_and_identifier_match(self):
        chunks = [dict(id="a", doc_id="lists", section=1, text="list sort 리스트 정렬"),
                  dict(id="b", doc_id="lists", section=1, text="sort None 반환"),
                  dict(id="c", doc_id="lists", section=1, text="리스트 설명"),
                  dict(id="d", doc_id="lists", section=2, text="sorted 새 리스트 정렬")]
        ids, _, _ = hybrid_rank("list.sort sorted 반환", chunks, ["a", "b", "c"], 3)
        self.assertIn("a", ids)
        self.assertIn("b", ids)
        self.assertIn("d", ids)


class PersistenceTests(unittest.TestCase):
    def test_cosine_query_returns_metadata_after_reopen(self):
        with tempfile.TemporaryDirectory() as folder:
            db = chromadb.PersistentClient(path=folder, settings=Settings(anonymized_telemetry=False))
            col = db.create_collection("fixture", embedding_function=None,
                                       configuration={"hnsw": {"space": "cosine"}})
            col.add(ids=["a", "b"], documents=["리스트", "예외"],
                    embeddings=[[1., 0.], [0., 1.]], metadatas=[{"url": "source-a"}, {"url": "source-b"}])
            reopened = chromadb.PersistentClient(path=folder, settings=Settings(anonymized_telemetry=False))
            got = reopened.get_collection("fixture", embedding_function=None).query(
                query_embeddings=[[1., 0.]], n_results=1, include=["documents", "metadatas", "distances"])
            self.assertEqual(got["ids"][0], ["a"])
            self.assertEqual(got["metadatas"][0][0]["url"], "source-a")


if __name__ == "__main__":
    unittest.main()
