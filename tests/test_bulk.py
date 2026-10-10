import csv
import gzip
import io
import json
import zipfile

from countrycode.real.bulk import brreg_country_values, ch_country_values, register_code_labels


def test_brreg_values_with_codes(tmp_path):
    data = [  # _fixture: fictitious records
        {"navn": "X", "forretningsadresse": {"land": "Norge", "landkode": "NO"}},
        {
            "navn": "Y",
            "forretningsadresse": {"land": "Storbritannia", "landkode": "GB"},
            "postadresse": {"land": "Storbritannia", "landkode": "GB"},
        },
        {"navn": "Z", "utenlandskRegisterAdresse": {"land": "Untied Kingdom"}},
    ]
    p = tmp_path / "e.json.gz"
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    vals = brreg_country_values(p)
    assert vals[("forretningsadresse.land", "Storbritannia")]["count"] == 1
    assert vals[("forretningsadresse.land", "Storbritannia")]["codes"] == {"GB": 1}
    assert vals[("utenlandskRegisterAdresse.land", "Untied Kingdom")]["codes"] == {}
    labels = register_code_labels(vals)
    assert labels[("postadresse.land", "Storbritannia")] == "GB"
    assert ("utenlandskRegisterAdresse.land", "Untied Kingdom") not in labels


def test_ch_values(tmp_path):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["CompanyName", " CompanyNumber", "RegAddress.Country", "CountryOfOrigin"])
    w.writerow(["A LTD", "1", "ENGLAND", "United Kingdom"])
    w.writerow(["B LTD", "2", "ENGLAND", "United Kingdom"])
    w.writerow(["C LTD", "3", "UNTIED KINGDOM", "Jersey"])
    p = tmp_path / "ch.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("BasicCompanyData.csv", buf.getvalue())
    vals = ch_country_values(p)
    assert vals[("RegAddress.Country", "ENGLAND")] == 2 and vals[("CountryOfOrigin", "Jersey")] == 1
