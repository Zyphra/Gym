# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text-layer extraction for the Anthropic ingress ``document`` block (``nemo_gym.pdf_text``)."""

import base64
import zlib
from typing import Dict, List, Optional

import pytest

from nemo_gym.pdf_text import PdfTextError, extract_pdf_text


def build_pdf(
    pages: List[bytes], fonts: Dict[str, bytes], extra: Optional[List[bytes]] = None, a85: bool = False
) -> bytes:
    """A small, valid PDF: one content stream per page, shared font resources, a real xref table.

    ``fonts`` maps a resource name to its font dictionary (object numbers from 100 up are free for
    ``extra`` objects such as a ToUnicode stream). With ``a85`` the content streams use
    ``[/ASCII85Decode /FlateDecode]``, as ReportLab writes them.
    """
    objects: Dict[int, bytes] = {}
    font_refs = []
    for i, (name, body) in enumerate(sorted(fonts.items())):
        objects[10 + i] = body
        font_refs.append(f"/{name} {10 + i} 0 R".encode())
    kids = []
    for i, content in enumerate(pages):
        page_num, content_num = 20 + 2 * i, 21 + 2 * i
        if a85:
            data = base64.a85encode(zlib.compress(content)) + b"~>"
            stream_dict = b"/Filter [ /ASCII85Decode /FlateDecode ] /Length %d" % len(data)
        else:
            data = content
            stream_dict = b"/Length %d" % len(data)
        objects[content_num] = b"<< " + stream_dict + b" >>\nstream\n" + data + b"\nendstream"
        objects[page_num] = b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R >>" % content_num
        kids.append(b"%d 0 R" % page_num)
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[2] = (
        b"<< /Type /Pages /Kids ["
        + b" ".join(kids)
        + b"] /Count %d" % len(pages)
        + b" /Resources << /Font << "
        + b" ".join(font_refs)
        + b" >> >> >>"
    )
    for i, body in enumerate(extra or []):
        objects[100 + i] = body
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += b"%d 0 obj\n" % num + objects[num] + b"\nendobj\n"
    size = max(objects) + 1
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % size
    for num in range(1, size):
        out += b"%010d 00000 n \n" % offsets[num] if num in offsets else b"0000000000 65535 f \n"
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (size, xref)
    return bytes(out)


HELVETICA = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
# ReportLab's CJK font: a Type0 font over the predefined UCS-2 CMap, so its codes are UTF-16BE.
ST_SONG = (
    b"<< /Type /Font /Subtype /Type0 /BaseFont /STSong-Light /Encoding /UniGB-UCS2-H "
    b"/DescendantFonts [ << /Type /Font /Subtype /CIDFontType0 /BaseFont /STSong-Light "
    b"/CIDSystemInfo << /Registry (Adobe) /Ordering (GB1) /Supplement 0 >> >> ] >>"
)


def ucs2(text: str) -> bytes:
    return b"<" + text.encode("utf-16-be").hex().encode() + b">"


def test_latin_text_lines_follow_text_positioning() -> None:
    page = (
        b"BT /F1 10 Tf 72 742 Td (Invoice CBS-2604-12) Tj 0 -14 Td (Amount: 2,310.40 USD) Tj "
        b"T* (Caf\\351 \\(paid\\)) Tj ET"
    )
    text, pages = extract_pdf_text(build_pdf([page], {"F1": HELVETICA}))
    assert pages == 1
    assert text == "Invoice CBS-2604-12\nAmount: 2,310.40 USD\nCafé (paid)"


def test_reportlab_style_cjk_page_with_ascii85_flate_streams() -> None:
    page = (
        b"BT /F2 12 Tf 1 0 0 1 72 700 Tm " + ucs2("采购合同") + b" Tj ET\n"
        b"BT /F2 10 Tf 1 0 0 1 72 680 Tm " + ucs2("合同总价款为 ") + b" Tj /F1 10 Tf (2,237,400.10) Tj ET\n"
        b"BT /F2 10 Tf 1 0 0 1 300 680 Tm " + ucs2("元") + b" Tj ET"
    )
    text, _ = extract_pdf_text(build_pdf([page], {"F1": HELVETICA, "F2": ST_SONG}, a85=True))
    # A new Tm on the same baseline is a horizontal move (a space), a lower one a new line.
    assert text == "采购合同\n合同总价款为 2,237,400.10 元"


def test_to_unicode_cmap_and_tj_spacing_across_pages() -> None:
    cmap = (
        b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        b"1 begincodespacerange <0000> <FFFF> endcodespacerange\n"
        b"2 beginbfchar <0001> <0048> <0002> <0069> endbfchar\n"
        b"1 beginbfrange <0010> <0012> <0061> endbfrange\n"
        b"endcmap CMapName currentdict /CMap defineresource pop end end"
    )
    to_unicode = b"<< /Length %d >>\nstream\n" % len(cmap) + cmap + b"\nendstream"
    identity = b"<< /Type /Font /Subtype /Type0 /BaseFont /Embedded /Encoding /Identity-H /ToUnicode 100 0 R >>"
    first = b"BT /F3 12 Tf 72 700 Td [<00010002> -600 <001000110012>] TJ ET"
    second = b"BT /F3 12 Tf 72 700 Td [<0012> 120 <0011>] TJ ET"
    text, pages = extract_pdf_text(build_pdf([first, second], {"F3": identity}, extra=[to_unicode]))
    assert pages == 2
    assert text == "Hi abc\n\f\ncb"  # a wide TJ gap is a space; pages are form-feed separated


def test_extraction_is_deterministic() -> None:
    pdf = build_pdf([b"BT /F1 9 Tf 10 10 Td (same bytes, same text) Tj ET"], {"F1": HELVETICA})
    assert extract_pdf_text(pdf) == extract_pdf_text(bytes(pdf)) == ("same bytes, same text", 1)


def test_a_page_without_text_returns_empty_text() -> None:
    text, pages = extract_pdf_text(build_pdf([b"0 0 m 100 100 l S"], {"F1": HELVETICA}))
    assert (text, pages) == ("", 1)


@pytest.mark.parametrize(
    "data",
    [b"not a pdf", b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n", b"%PDF-1.4\ntrailer << /Encrypt 5 0 R >>\n"],
)
def test_unreadable_files_raise_pdf_text_error(data: bytes) -> None:
    with pytest.raises(PdfTextError):
        extract_pdf_text(data)
