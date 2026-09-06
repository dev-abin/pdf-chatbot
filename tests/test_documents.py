from pdf_chatbot.documents import Page, split_pages
from pdf_chatbot.retrieval import rank_chunks


def test_chunks_keep_source_and_page_number():
    chunks = split_pages(
        [Page(number=4, text="A" * 30)],
        "guide.pdf",
        chunk_size=20,
        overlap=5,
    )

    assert [chunk.page_number for chunk in chunks] == [4, 4]
    assert all(chunk.source_name == "guide.pdf" for chunk in chunks)


def test_retrieval_ranks_matching_chunk_first():
    pages = [
        Page(number=1, text="Python uses indentation to define code blocks."),
        Page(number=2, text="PDF pages can contain images."),
    ]
    chunks = split_pages(pages, "notes.pdf", chunk_size=200, overlap=20)

    ranked = rank_chunks(chunks, "How does Python define code blocks?")

    assert ranked[0].chunk.page_number == 1
