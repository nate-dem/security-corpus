import tempfile
from pathlib import Path

from ingest.readers import read


def test_read_xml_streams_elements():
    xml_content = b"""\
<?xml version="1.0" encoding="UTF-8"?>
<Attack_Pattern_Catalog xmlns="http://capec.mitre.org/capec-3">
  <Attack_Patterns>
    <Attack_Pattern ID="1" Name="Pattern One" Abstraction="Meta" Status="Stable">
      <Description>First pattern</Description>
    </Attack_Pattern>
    <Attack_Pattern ID="2" Name="Pattern Two" Abstraction="Standard" Status="Draft">
      <Description>Second pattern</Description>
    </Attack_Pattern>
  </Attack_Patterns>
</Attack_Pattern_Catalog>
"""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        f.write(xml_content)
        f.flush()
        path = Path(f.name)

    tag = "{http://capec.mitre.org/capec-3}Attack_Pattern"
    # Elements are cleared after yield, so capture attributes during iteration
    results = [(elem.get("ID"), elem.get("Name")) for elem in read(path, xml_tag=tag)]

    assert len(results) == 2
    assert results[0] == ("1", "Pattern One")
    assert results[1] == ("2", "Pattern Two")

    path.unlink()


def test_read_xml_requires_xml_tag():
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        f.write(b"<root/>")
        f.flush()
        path = Path(f.name)

    try:
        list(read(path))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "xml_tag" in str(e)
    finally:
        path.unlink()
