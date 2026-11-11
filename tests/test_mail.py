import io
from email.message import EmailMessage
from types import SimpleNamespace as NS

import pytest

from citerag import config, layout, mail
from citerag.generator import location
from citerag.loaders import load_file

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@pytest.fixture(autouse=True)
def parse_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PARSE_CACHE_DIR", tmp_path / "parsed")


def budget_docx() -> bytes:
    from docx import Document

    doc = Document()
    doc.add_heading("Setup budget", level=1)
    t = doc.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Item", "Cost (NPR)"
    t.cell(1, 0).text, t.cell(1, 1).text = "IT equipment", "600,000"
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_eml(path, attach=(), forward=None, body="The landlord agreed to NPR 95,000 per month."):
    msg = EmailMessage()
    msg["Subject"] = "Pokhara office lease"
    msg["From"] = "Anita Sharma <anita@northwind.example>"
    msg["To"] = "ramesh@northwind.example"
    msg["Date"] = "Wed, 04 Nov 2026 10:15:00 +0545"
    msg.set_content(body)
    msg.add_alternative(f"<html><body><p>{body}</p><img src='cid:logo'></body></html>", subtype="html")
    msg.get_payload()[1].add_related(b"\x89PNG" + b"0" * 500, maintype="image", subtype="png", cid="<logo>",
                                     filename="logo.png")  # an inline logo
    for name, data, (maintype, subtype) in attach:
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=name)
    if forward is not None:
        msg.add_attachment(forward)  # message/rfc822
    path.write_bytes(bytes(msg))
    return path


def test_attachments_skip_inline_and_small_images_and_open_forwards(tmp_path):
    inner = EmailMessage()
    inner["Subject"] = "Original"
    inner.set_content("Earlier message.")
    inner.add_attachment(b"%PDF-1.4", maintype="application", subtype="pdf", filename="lease.pdf")
    path = make_eml(tmp_path / "m.eml", attach=[
        ("budget.docx", b"PK..", ("application", DOCX.split("/")[1])),
        ("icon.png", b"\x89PNG" + b"0" * 100, ("image", "png")),  # tiny: a signature icon
        ("scan.png", b"\x89PNG" + b"0" * 40_000, ("image", "png")),  # big enough to be a real scan
        ("../../evil.txt", b"x", ("text", "plain")),
    ], forward=inner)
    names = [a.name for a in mail.attachments(path)]
    # No logo.png or icon.png; the forwarded email's text is in the body, its file comes out.
    assert names == ["budget.docx", "scan.png", "evil.txt", "lease.pdf"]


def test_safe_names_never_leave_the_folder():
    assert mail.safe_name("../../etc/passwd", 1) == "passwd"
    assert mail.safe_name("C:\\Users\\x\\report.pdf", 2) == "report.pdf"
    assert mail.safe_name("", 3, ".eml") == "attachment-3.eml"
    assert mail.safe_name("bad\x00name?.pdf", 4) == "bad_name_.pdf"


def test_outlook_msg_attachments(monkeypatch, tmp_path):
    fake = NS(attachments=[NS(file_name="plan.docx", file_bytes=b"PK"), NS(file_name=None, file_bytes=b"%PDF")])
    import oxmsg

    monkeypatch.setattr(oxmsg.Message, "load", staticmethod(lambda path: fake))
    (tmp_path / "m.msg").write_bytes(b"\xd0\xcf")
    assert [a.name for a in mail.attachments(tmp_path / "m.msg")] == ["plan.docx", "attachment-2"]


@pytest.mark.skipif(not layout.available(), reason="docling not installed")
def test_email_is_read_with_its_attachments(tmp_path, monkeypatch):
    deepest = EmailMessage()
    deepest["Subject"] = "Older thread"
    deepest.set_content("Two levels down.")
    inner = EmailMessage()
    inner["Subject"] = "Lease draft"
    inner.set_content("The deposit is three months' rent.")
    inner.add_attachment(b"Nested attachment text.", maintype="text", subtype="plain", filename="deep.txt")
    inner.add_attachment(deepest)
    path = make_eml(tmp_path / "lease.eml", attach=[
        ("budget.docx", budget_docx(), tuple(DOCX.split("/"))),
        ("contacts.csv", b"name,phone\n", ("text", "csv")),
    ], forward=inner)
    [page] = load_file(path)
    text = page.text
    assert text.startswith("# Pokhara office lease\n\nFrom: Anita Sharma <anita@northwind.example>")
    assert "The landlord agreed to NPR 95,000 per month." in text
    assert ("## Attachment: budget.docx\n\n### budget.docx: Setup budget\n\n"
            "- Item: IT equipment; Cost (NPR): 600,000") in text
    # A forwarded email's text is in the body (Docling), once; its files are read as attachments.
    assert text.count("The deposit is three months' rent.") == 1 and text.count("Two levels down.") == 1
    assert "## Attachment: deep.txt\n\nNested attachment text." in text
    assert "## Other attachments (not read)\n\n- contacts.csv" in text
    assert "logo.png" not in text
    assert location(page.source, 1, 1) == "lease.eml"  # cited by name and section, not "p. 1"

    monkeypatch.setattr(config, "EMAIL_ATTACHMENTS", False)
    [page] = load_file(path)
    assert "Attachment" not in page.text and "95,000" in page.text
