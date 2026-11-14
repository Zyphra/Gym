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
"""Deterministic, standard-library-only extraction of a PDF's text layer.

The Anthropic ingress uses this to hand a text-only policy the text of a PDF ``document``
block (what Claude Code's Read tool sends for a PDF). It reads the text layer only: no OCR,
no images, no layout analysis. It supports what text-layer PDFs commonly use: classic and
cross-reference-stream files, object streams, the Flate/ASCII85/ASCIIHex/RunLength filters,
``ToUnicode`` CMaps, the predefined UCS-2/UTF-16 CJK CMaps (e.g. ``UniGB-UCS2-H``) and the
standard single-byte encodings. Lines follow the content-stream order; a vertical move starts
a new line and a horizontal move or a wide ``TJ`` gap inserts a space.

The same bytes always give the same text, independent of what else is installed.
"""

import base64
import re
import zlib
from typing import Any, Dict, List, Optional, Tuple


class PdfTextError(ValueError):
    """The PDF cannot be read (malformed, encrypted, or holds no readable text layer)."""


class _Name(str):
    pass


class _Ref(tuple):
    pass


class _Op(str):
    pass


_WHITESPACE = b"\x00\t\n\x0c\r "
_DELIMITERS = b"()<>[]{}/%"
_MAX_FORM_DEPTH = 8

# Predefined CMaps whose codes are UCS-2 / UTF-16BE code units (PDF 32000-1, 9.7.5.2).
_UNICODE_CMAP_SUFFIXES = ("-UCS2-H", "-UCS2-V", "-UTF16-H", "-UTF16-V")
_SINGLE_BYTE_ENCODINGS = {
    "WinAnsiEncoding": "cp1252",
    "MacRomanEncoding": "mac_roman",
    "StandardEncoding": "cp1252",
    "PDFDocEncoding": "latin-1",
}


class _Lexer:
    def __init__(self, data: bytes, pos: int = 0) -> None:
        self.data = data
        self.pos = pos

    def _skip_space(self) -> None:
        data, n = self.data, len(self.data)
        while self.pos < n:
            c = data[self.pos]
            if c in _WHITESPACE:
                self.pos += 1
            elif c == 0x25:  # % comment
                while self.pos < n and data[self.pos] not in b"\r\n":
                    self.pos += 1
            else:
                return

    def token(self) -> Any:
        """Next token; ``None`` at end of data. Operators and keywords come back as ``_Op``."""
        self._skip_space()
        data, n = self.data, len(self.data)
        if self.pos >= n:
            return None
        c = data[self.pos : self.pos + 1]
        if c == b"(":
            return self._literal_string()
        if c == b"<":
            if data[self.pos + 1 : self.pos + 2] == b"<":
                self.pos += 2
                return _Op("<<")
            return self._hex_string()
        if c == b">" and data[self.pos + 1 : self.pos + 2] == b">":
            self.pos += 2
            return _Op(">>")
        if c in (b"[", b"]", b"{", b"}"):
            self.pos += 1
            return _Op(c.decode())
        if c == b"/":
            self.pos += 1
            start = self.pos
            while self.pos < n and data[self.pos] not in _WHITESPACE and data[self.pos] not in _DELIMITERS:
                self.pos += 1
            raw = data[start : self.pos]
            name = re.sub(rb"#([0-9A-Fa-f]{2})", lambda m: bytes([int(m.group(1), 16)]), raw)
            return _Name(name.decode("latin-1"))
        start = self.pos
        while self.pos < n and data[self.pos] not in _WHITESPACE and data[self.pos] not in _DELIMITERS:
            self.pos += 1
        if self.pos == start:  # a stray delimiter such as ')' or '>'
            self.pos += 1
            return _Op(data[start : self.pos].decode("latin-1"))
        word = data[start : self.pos]
        try:
            return int(word)
        except ValueError:
            pass
        try:
            return float(word)
        except ValueError:
            return _Op(word.decode("latin-1"))

    def _literal_string(self) -> bytes:
        data, n = self.data, len(self.data)
        self.pos += 1
        depth = 1
        out = bytearray()
        escapes = {ord("n"): 10, ord("r"): 13, ord("t"): 9, ord("b"): 8, ord("f"): 12}
        while self.pos < n:
            c = data[self.pos]
            self.pos += 1
            if c == 0x5C:  # backslash
                if self.pos >= n:
                    break
                e = data[self.pos]
                self.pos += 1
                if e in escapes:
                    out.append(escapes[e])
                elif 0x30 <= e <= 0x37:
                    digits = bytes([e])
                    while len(digits) < 3 and self.pos < n and 0x30 <= data[self.pos] <= 0x37:
                        digits += data[self.pos : self.pos + 1]
                        self.pos += 1
                    out.append(int(digits, 8) & 0xFF)
                elif e == 0x0D:  # line continuation
                    if self.pos < n and data[self.pos] == 0x0A:
                        self.pos += 1
                elif e == 0x0A:
                    pass
                else:
                    out.append(e)
            elif c == 0x28:
                depth += 1
                out.append(c)
            elif c == 0x29:
                depth -= 1
                if depth == 0:
                    break
                out.append(c)
            else:
                out.append(c)
        return bytes(out)

    def _hex_string(self) -> bytes:
        end = self.data.find(b">", self.pos)
        if end < 0:
            end = len(self.data)
        digits = re.sub(rb"[^0-9A-Fa-f]", b"", self.data[self.pos + 1 : end])
        self.pos = end + 1
        if len(digits) % 2:
            digits += b"0"
        return bytes.fromhex(digits.decode())

    def obj(self, tok: Any = None) -> Any:
        """Parse one object (resolving ``n g R`` into ``_Ref``)."""
        if tok is None:
            tok = self.token()
        if isinstance(tok, _Op):
            if tok == "<<":
                out: Dict[str, Any] = {}
                while True:
                    key = self.token()
                    if key is None or key == ">>":
                        return out
                    if isinstance(key, _Name):
                        out[str(key)] = self.obj()
            if tok == "[":
                items: List[Any] = []
                while True:
                    t = self.token()
                    if t is None or t == "]":
                        return items
                    items.append(self.obj(t))
            if tok == "true":
                return True
            if tok == "false":
                return False
            if tok == "null":
                return None
            return tok
        if isinstance(tok, int):
            save = self.pos
            t2 = self.token()
            if isinstance(t2, int):
                t3 = self.token()
                if t3 == "R":
                    return _Ref((tok, t2))
            self.pos = save
        return tok


def _apply_filters(raw: bytes, stream_dict: Dict[str, Any], resolve) -> bytes:
    filters = resolve(stream_dict.get("Filter"))
    params = resolve(stream_dict.get("DecodeParms"))
    if filters is None:
        return raw
    if not isinstance(filters, list):
        filters = [filters]
        params = [params]
    elif not isinstance(params, list):
        params = [params] * len(filters)
    data = raw
    for name, parm in zip(filters, params):
        name = str(resolve(name))
        parm = resolve(parm) or {}
        if name in ("FlateDecode", "Fl"):
            try:
                data = zlib.decompress(data)
            except zlib.error:
                data = zlib.decompressobj().decompress(data)
            data = _png_predictor(data, parm)
        elif name in ("ASCII85Decode", "A85"):
            body = data.strip()
            if body.startswith(b"<~"):
                body = body[2:]
            end = body.find(b"~>")
            if end >= 0:
                body = body[:end]
            data = base64.a85decode(re.sub(rb"\s", b"", body))
        elif name in ("ASCIIHexDecode", "AHx"):
            end = data.find(b">")
            digits = re.sub(rb"[^0-9A-Fa-f]", b"", data if end < 0 else data[:end])
            if len(digits) % 2:
                digits += b"0"
            data = bytes.fromhex(digits.decode())
        elif name in ("RunLengthDecode", "RL"):
            data = _run_length(data)
        else:
            raise PdfTextError(f"unsupported stream filter {name}")
    return data


def _png_predictor(data: bytes, parm: Dict[str, Any]) -> bytes:
    predictor = parm.get("Predictor", 1) if isinstance(parm, dict) else 1
    if not isinstance(predictor, int) or predictor < 10:
        return data
    columns = parm.get("Columns", 1)
    colors = parm.get("Colors", 1)
    bpc = parm.get("BitsPerComponent", 8)
    bpp = max(1, colors * bpc // 8)
    row_len = (columns * colors * bpc + 7) // 8
    out = bytearray()
    prev = bytearray(row_len)
    for i in range(0, len(data), row_len + 1):
        kind = data[i]
        row = bytearray(data[i + 1 : i + 1 + row_len].ljust(row_len, b"\0"))
        for j in range(row_len):
            left = row[j - bpp] if j >= bpp else 0
            up = prev[j]
            up_left = prev[j - bpp] if j >= bpp else 0
            if kind == 1:
                row[j] = (row[j] + left) & 0xFF
            elif kind == 2:
                row[j] = (row[j] + up) & 0xFF
            elif kind == 3:
                row[j] = (row[j] + (left + up) // 2) & 0xFF
            elif kind == 4:
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                pred = left if pa <= pb and pa <= pc else (up if pb <= pc else up_left)
                row[j] = (row[j] + pred) & 0xFF
        out += row
        prev = row
    return bytes(out)


def _run_length(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data):
        length = data[i]
        if length == 128:
            break
        if length < 128:
            out += data[i + 1 : i + 2 + length]
            i += length + 2
        else:
            out += data[i + 1 : i + 2] * (257 - length)
            i += 2
    return bytes(out)


class _Document:
    def __init__(self, data: bytes) -> None:
        if not data.lstrip()[:5] == b"%PDF-":
            raise PdfTextError("not a PDF (missing %PDF- header)")
        self.data = data
        self.objects: Dict[int, Any] = {}
        self.streams: Dict[int, Tuple[Dict[str, Any], bytes]] = {}
        self.trailer: Dict[str, Any] = {}
        self._decoded: Dict[int, bytes] = {}
        self._scan()

    def _scan(self) -> None:
        data = self.data
        for m in re.finditer(rb"(?<![0-9])(\d+)\s+(\d+)\s+obj\b", data):
            num = int(m.group(1))
            lex = _Lexer(data, m.end())
            try:
                value = lex.obj()
            except (ValueError, IndexError):
                continue
            after = _Lexer(data, lex.pos)
            after._skip_space()
            if isinstance(value, dict) and data.startswith(b"stream", after.pos):
                start = after.pos + len(b"stream")
                if data[start : start + 2] == b"\r\n":
                    start += 2
                elif data[start : start + 1] in (b"\n", b"\r"):
                    start += 1
                length = value.get("Length")
                end = start + length if isinstance(length, int) else -1
                if end < 0 or not re.match(rb"\s*endstream", data[end : end + 20]):
                    end = data.find(b"endstream", start)
                    if end < 0:
                        continue
                    raw = data[start:end].rstrip(b"\r\n")
                else:
                    raw = data[start:end]
                self.streams[num] = (value, raw)
                self.objects[num] = value
                if value.get("Type") == "XRef":
                    self._merge_trailer(value)
            else:
                self.objects[num] = value
        for m in re.finditer(rb"trailer\s*<<", data):
            lex = _Lexer(data, m.end() - 2)
            try:
                trailer = lex.obj()
            except (ValueError, IndexError):
                continue
            if isinstance(trailer, dict):
                self._merge_trailer(trailer)
        for num, (sdict, _raw) in list(self.streams.items()):
            if sdict.get("Type") == "ObjStm":
                self._expand_object_stream(num, sdict)
        if self.trailer.get("Encrypt") is not None:
            raise PdfTextError("encrypted PDF")

    def _merge_trailer(self, trailer: Dict[str, Any]) -> None:
        for key in ("Root", "Encrypt", "Info"):
            if key in trailer:
                self.trailer[key] = trailer[key]

    def _expand_object_stream(self, num: int, sdict: Dict[str, Any]) -> None:
        try:
            body = self.stream_data(num)
        except PdfTextError:
            return
        count = self.resolve(sdict.get("N")) or 0
        first = self.resolve(sdict.get("First")) or 0
        header = _Lexer(body[:first])
        offsets = []
        for _ in range(count):
            obj_num, offset = header.token(), header.token()
            if not isinstance(obj_num, int) or not isinstance(offset, int):
                break
            offsets.append((obj_num, offset))
        for obj_num, offset in offsets:
            if obj_num in self.objects:
                continue  # a later classic definition wins
            try:
                self.objects[obj_num] = _Lexer(body, first + offset).obj()
            except (ValueError, IndexError):
                continue

    def resolve(self, value: Any, depth: int = 0) -> Any:
        while isinstance(value, _Ref) and depth < 32:
            value = self.objects.get(value[0])
            depth += 1
        return value

    def stream_data(self, num: int) -> bytes:
        if num not in self._decoded:
            sdict, raw = self.streams[num]
            try:
                self._decoded[num] = _apply_filters(raw, sdict, self.resolve)
            except (zlib.error, ValueError) as exc:
                raise PdfTextError(f"cannot decode stream {num}: {exc}") from exc
        return self._decoded[num]

    def stream_of(self, value: Any) -> Optional[bytes]:
        if isinstance(value, _Ref) and value[0] in self.streams:
            return self.stream_data(value[0])
        return None

    def pages(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Pages in document order, each with its inherited resources."""
        root = self.resolve(self.trailer.get("Root"))
        if not isinstance(root, dict):
            root = next((o for o in self.objects.values() if isinstance(o, dict) and o.get("Type") == "Catalog"), None)
        out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        seen: set = set()

        def walk(node_ref: Any, inherited: Dict[str, Any]) -> None:
            key = node_ref[0] if isinstance(node_ref, _Ref) else id(node_ref)
            if key in seen:
                return
            seen.add(key)
            node = self.resolve(node_ref)
            if not isinstance(node, dict):
                return
            resources = self.resolve(node.get("Resources"))
            if not isinstance(resources, dict):
                resources = inherited
            kids = self.resolve(node.get("Kids"))
            if node.get("Type") == "Pages" or isinstance(kids, list):
                for kid in kids or []:
                    walk(kid, resources)
            else:
                out.append((node, resources))

        if isinstance(root, dict):
            walk(root.get("Pages"), {})
        if not out:  # no usable page tree: take page objects in object-number order
            for num in sorted(self.objects):
                obj = self.objects[num]
                if isinstance(obj, dict) and obj.get("Type") == "Page":
                    out.append((obj, self.resolve(obj.get("Resources")) or {}))
        return out


def _parse_cmap(data: bytes) -> Tuple[Dict[bytes, str], List[int]]:
    """``ToUnicode`` CMap -> (code bytes -> text, code byte lengths in use)."""
    mapping: Dict[bytes, str] = {}
    lengths: set = set()

    def text_of(raw: bytes) -> str:
        return raw.decode("utf-16-be", errors="replace") if len(raw) >= 2 else raw.decode("latin-1")

    for m in re.finditer(rb"begincodespacerange(.*?)endcodespacerange", data, re.S):
        for lo in re.findall(rb"<([0-9A-Fa-f\s]+)>\s*<[0-9A-Fa-f\s]+>", m.group(1)):
            lengths.add(len(re.sub(rb"\s", b"", lo)) // 2)
    for m in re.finditer(rb"beginbfchar(.*?)endbfchar", data, re.S):
        lex = _Lexer(m.group(1))
        while True:
            src, dst = lex.token(), lex.token()
            if not isinstance(src, bytes) or not isinstance(dst, (bytes, _Name)):
                break
            mapping[src] = text_of(dst) if isinstance(dst, bytes) else str(dst)
            lengths.add(len(src))
    for m in re.finditer(rb"beginbfrange(.*?)endbfrange", data, re.S):
        lex = _Lexer(m.group(1))
        while True:
            lo, hi = lex.token(), lex.token()
            if not isinstance(lo, bytes) or not isinstance(hi, bytes):
                break
            dst = lex.obj()
            width = len(lo)
            lengths.add(width)
            start, stop = int.from_bytes(lo, "big"), int.from_bytes(hi, "big")
            if stop - start > 0xFFFF:
                continue
            if isinstance(dst, list):
                for i, d in enumerate(dst[: stop - start + 1]):
                    if isinstance(d, bytes):
                        mapping[(start + i).to_bytes(width, "big")] = text_of(d)
            elif isinstance(dst, bytes) and dst:
                base = int.from_bytes(dst, "big")
                for i in range(stop - start + 1):
                    mapping[(start + i).to_bytes(width, "big")] = text_of((base + i).to_bytes(max(len(dst), 2), "big"))
    return mapping, sorted(lengths) or [1]


class _Font:
    def __init__(self, doc: _Document, font: Dict[str, Any]) -> None:
        self.cmap: Optional[Dict[bytes, str]] = None
        self.code_lengths = [1]
        self.codec: Optional[str] = "cp1252"
        self.differences: Dict[int, str] = {}
        subtype = font.get("Subtype")
        encoding = doc.resolve(font.get("Encoding"))
        to_unicode = doc.stream_of(font.get("ToUnicode")) if font.get("ToUnicode") is not None else None
        if subtype == "Type0":
            self.code_lengths = [2]
            self.codec = None
            if isinstance(encoding, _Name) and str(encoding).endswith(_UNICODE_CMAP_SUFFIXES):
                self.codec = "utf-16-be"
        elif isinstance(encoding, _Name):
            self.codec = _SINGLE_BYTE_ENCODINGS.get(str(encoding), "cp1252")
        elif isinstance(encoding, dict):
            base = encoding.get("BaseEncoding")
            self.codec = _SINGLE_BYTE_ENCODINGS.get(str(base), "cp1252") if base else "cp1252"
            code = 0
            for item in doc.resolve(encoding.get("Differences")) or []:
                if isinstance(item, int):
                    code = item
                elif isinstance(item, _Name):
                    glyph = _glyph_to_text(str(item))
                    if glyph is not None:
                        self.differences[code] = glyph
                    code += 1
        if to_unicode:
            self.cmap, self.code_lengths = _parse_cmap(to_unicode)

    def decode(self, raw: bytes) -> str:
        if self.cmap is not None:
            out = []
            i = 0
            while i < len(raw):
                for width in self.code_lengths:
                    code = raw[i : i + width]
                    if code in self.cmap:
                        out.append(self.cmap[code])
                        i += width
                        break
                else:
                    if self.codec == "utf-16-be" and i + 1 < len(raw):
                        out.append(raw[i : i + 2].decode("utf-16-be", errors="replace"))
                        i += 2
                    else:
                        i += self.code_lengths[0]
            return "".join(out)
        if self.codec is None:  # a composite font with no Unicode mapping: codes are glyph ids
            return ""
        if self.codec == "utf-16-be":
            return raw.decode("utf-16-be", errors="replace")
        if self.differences:
            return "".join(self.differences.get(b, bytes([b]).decode(self.codec, errors="replace")) for b in raw)
        return raw.decode(self.codec, errors="replace")


_GLYPH_NAMES = {
    "space": " ",
    "quotesingle": "'",
    "quotedbl": '"',
    "bullet": "•",
    "endash": "–",
    "emdash": "—",
    "quoteleft": "‘",
    "quoteright": "’",
    "quotedblleft": "“",
    "quotedblright": "”",
    "fi": "fi",
    "fl": "fl",
    "ellipsis": "…",
    "hyphen": "-",
    "period": ".",
    "comma": ",",
    "colon": ":",
    "semicolon": ";",
    "parenleft": "(",
    "parenright": ")",
    "slash": "/",
    "percent": "%",
    "ampersand": "&",
    "dollar": "$",
    "Euro": "€",
}
_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _glyph_to_text(name: str) -> Optional[str]:
    if len(name) == 1:
        return name
    if name in _GLYPH_NAMES:
        return _GLYPH_NAMES[name]
    if name in _DIGITS:
        return str(_DIGITS.index(name))
    m = re.fullmatch(r"uni([0-9A-Fa-f]{4})+", name)
    if m:
        return "".join(chr(int(name[i : i + 4], 16)) for i in range(3, len(name), 4))
    m = re.fullmatch(r"u([0-9A-Fa-f]{4,6})", name)
    if m:
        return chr(int(m.group(1), 16))
    return None


class _TextWriter:
    def __init__(self) -> None:
        self.parts: List[str] = []
        self.pending = ""

    def move(self, dy: float, dx: float) -> None:
        if abs(dy) > 0.01:
            self.pending = "\n"
        elif abs(dx) > 0.01 and self.pending != "\n":
            self.pending = " "

    def write(self, text: str) -> None:
        if not text:
            return
        if self.parts and self.pending:
            last = self.parts[-1][-1:]
            if self.pending == "\n" and last != "\n":
                self.parts.append("\n")
            elif self.pending == " " and not last.isspace() and not text[:1].isspace():
                self.parts.append(" ")
        self.pending = ""
        self.parts.append(text)

    def text(self) -> str:
        return "".join(self.parts)


def _run_content(doc: _Document, content: bytes, resources: Dict[str, Any], out: _TextWriter, depth: int) -> None:
    fonts_dict = doc.resolve(resources.get("Font")) if isinstance(resources, dict) else None
    xobjects = doc.resolve(resources.get("XObject")) if isinstance(resources, dict) else None
    fonts: Dict[str, _Font] = {}
    font: Optional[_Font] = None
    line_y = 0.0
    lex = _Lexer(content)
    operands: List[Any] = []
    while True:
        tok = lex.token()
        if tok is None:
            break
        if isinstance(tok, _Op) and tok not in ("<<", "[", "true", "false", "null"):
            op = str(tok)
            if op == "BI":  # inline image: skip its binary data
                end = re.search(rb"\sEI(?=[\s]|$)", content[lex.pos :])
                lex.pos = lex.pos + end.end() if end else len(content)
            elif op == "Tf" and len(operands) >= 2:
                name = str(operands[-2])
                if name not in fonts:
                    fdict = doc.resolve(fonts_dict.get(name)) if isinstance(fonts_dict, dict) else None
                    fonts[name] = _Font(doc, fdict) if isinstance(fdict, dict) else _Font(doc, {})
                font = fonts[name]
            elif op in ("Td", "TD") and len(operands) >= 2:
                dx, dy = operands[-2], operands[-1]
                if isinstance(dx, (int, float)) and isinstance(dy, (int, float)):
                    line_y += dy
                    out.move(dy, dx)
            elif op == "Tm" and len(operands) >= 6:
                y = operands[-1]
                if isinstance(y, (int, float)):
                    out.move(y - line_y, 1)
                    line_y = y
            elif op in ("T*", "'", '"'):
                out.move(1, 0)
                if op != "T*" and operands and isinstance(operands[-1], bytes) and font is not None:
                    out.write(font.decode(operands[-1]))
            elif op == "Tj" and operands and isinstance(operands[-1], bytes) and font is not None:
                out.write(font.decode(operands[-1]))
            elif op == "TJ" and operands and isinstance(operands[-1], list) and font is not None:
                for item in operands[-1]:
                    if isinstance(item, bytes):
                        out.write(font.decode(item))
                    elif isinstance(item, (int, float)) and item < -250:
                        out.move(0, 1)
            elif op == "Do" and operands and depth < _MAX_FORM_DEPTH and isinstance(xobjects, dict):
                ref = xobjects.get(str(operands[-1]))
                xobj = doc.resolve(ref)
                if isinstance(xobj, dict) and xobj.get("Subtype") == "Form":
                    data = doc.stream_of(ref)
                    if data:
                        form_resources = doc.resolve(xobj.get("Resources")) or resources
                        _run_content(doc, data, form_resources, out, depth + 1)
            operands = []
        else:
            operands.append(lex.obj(tok))


def extract_pdf_text(data: bytes) -> Tuple[str, int]:
    """Return ``(text, page_count)`` for the PDF's text layer.

    Pages are separated by a form feed. Raises ``PdfTextError`` when the file cannot be
    parsed or is encrypted. A PDF whose pages carry no text layer (e.g. a scan) returns
    empty text; the caller decides how to report that.
    """
    try:
        doc = _Document(data)
        pages = doc.pages()
        if not pages:
            raise PdfTextError("no pages found")
        texts = []
        for page, resources in pages:
            contents = page.get("Contents")
            refs = doc.resolve(contents)
            refs = refs if isinstance(refs, list) else [contents]
            streams = [doc.stream_of(r) for r in refs]
            out = _TextWriter()
            _run_content(doc, b"\n".join(s for s in streams if s), resources or {}, out, 0)
            texts.append("\n".join(line.rstrip() for line in out.text().splitlines()).strip("\n"))
    except PdfTextError:
        raise
    except (ValueError, IndexError, KeyError, TypeError, RecursionError, zlib.error) as exc:
        raise PdfTextError(f"malformed PDF: {type(exc).__name__}: {exc}") from exc
    return "\n\f\n".join(texts).strip(), len(pages)
