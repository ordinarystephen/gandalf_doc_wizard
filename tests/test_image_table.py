import sys; sys.path.insert(0, '.')
from unittest.mock import MagicMock, patch


def _make_mock_pixmap():
    """Minimal pixmap mock returning valid 1x1 white PNG bytes."""
    import struct, zlib
    def make_png():
        def chunk(name, data):
            c = name + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', 0)
        sig = b'\x89PNG\r\n\x1a\n'
        ihdr = chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
        raw = zlib.compress(b'\x00\xff\xff\xff')
        idat = chunk(b'IDAT', raw)
        iend = chunk(b'IEND', b'')
        return sig + ihdr + idat + iend
    pm = MagicMock()
    pm.tobytes.return_value = make_png()
    return pm


def test_no_table_returns_none():
    with patch("doc_qa.ingest.image_table._call_vision_api", return_value="NO_TABLE"):
        from doc_qa.ingest.image_table import extract_image_table
        result = extract_image_table(
            pixmap=_make_mock_pixmap(), page_number=1,
            bounding_box=(0, 0, 100, 100), filename="test.pdf",
        )
        assert result is None


def test_table_response_returns_chunk():
    md = "| Col1 | Col2 |\n|------|------|\n| A    | 1.0  |"
    with patch("doc_qa.ingest.image_table._call_vision_api", return_value=md):
        from doc_qa.ingest.image_table import extract_image_table
        chunk = extract_image_table(
            pixmap=_make_mock_pixmap(), page_number=2,
            bounding_box=(10, 20, 300, 400), filename="test.pdf",
        )
        assert chunk is not None
        assert chunk.content_type == "image_table"
        assert chunk.extraction_method == "vision_ocr_gpt4o"
        assert chunk.page_number == 2
        assert chunk.char_count == len(chunk.text)


def test_api_failure_returns_none():
    with patch("doc_qa.ingest.image_table._call_vision_api",
               side_effect=Exception("API down")):
        from doc_qa.ingest.image_table import extract_image_table
        result = extract_image_table(
            pixmap=_make_mock_pixmap(), page_number=1,
            bounding_box=(0, 0, 100, 100), filename="test.pdf",
        )
        assert result is None
