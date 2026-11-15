import pytest

from citerag.privacy import find_private, mask, mask_value, streaming_safe, summary

FOUND = [
    ("Call me on 9841234567 tomorrow.", "phone", "Call me on 98••••••67 tomorrow."),
    ("Mobile: +977-9851234567", "phone", "Mobile: +977-98••••••67"),
    ("Office: 01-4412345", "phone", "Office: 01-•••••45"),
    ("Pokhara office (061) 523456", "phone", "Pokhara office (06•) ••••56"),
    ("Tel. +44 20 7946 0958", "phone", "Tel. +44 2• •••• ••58"),
    ("मोबाइल: ९८४१२३४५६७", "phone", "मोबाइल: ९८••••••६७"),
    ("Citizenship No.: 27-01-71-12345", "id_number", "Citizenship No.: ••-••-••-•••45"),
    ("नागरिकता नं. २७-०१-७१-१२३४५", "id_number", "नागरिकता नं. ••-••-••-•••४५"),
    ("PAN: 301234567", "id_number", "PAN: •••••••67"),
    ("Passport No. PA1234567", "id_number", "Passport No. •••••••67"),
    ("Account no. 0012345678901", "bank_account", "Account no. ••••••••••901"),
    ("A/C: 01701070012345000001", "bank_account", "A/C: •••••••••••••••••001"),
    ("card 4111 1111 1111 1111", "card", "card •••• •••• •••• 1111"),
    ("IBAN GB82 WEST 1234 5698 7654 32", "bank_account", "IBAN •••• •••• •••• •••• •••4 32"),
]
NOT_PRIVATE = [
    "In 2017 the model scored 28.4 BLEU on 4.5 million sentence pairs.",
    "Rent is Rs. 1,250,000 per year.",
    "Meeting on 2026-11-11 at 10:30.",
    "ISBN 978-3-16-148410-0",
    "dimension 512 and 2048 heads; 65,000 tokens",
    "arXiv:1706.03762 and version 2.146.0",
    "The pan is hot.",
    "Revenue was NPR 9,800,000,000 in Q3 2081/82.",
    "Account managers approve requests.",
    "card 4111 1111 1111 1112",  # fails the Luhn checksum
]


@pytest.mark.parametrize("text, kind, masked", FOUND)
def test_private_values_are_found_and_masked(text, kind, masked):
    assert [f.kind for f in find_private(text)] == [kind]
    assert mask(text) == masked
    assert len(mask(text)) == len(text)  # offsets into the text stay valid


@pytest.mark.parametrize("text", NOT_PRIVATE)
def test_ordinary_numbers_are_left_alone(text):
    assert find_private(text) == [] and mask(text) == text


def test_several_values_in_one_answer():
    text = "Ring 9841234567 or 01-4412345; deposit to A/C 0012345678901."
    found = find_private(text)
    assert [f.kind for f in found] == ["phone", "phone", "bank_account"]
    assert summary(found) == "2 phone numbers and 1 bank account number"
    assert "9841234567" not in mask(text) and "0012345678901" not in mask(text)


def test_a_number_still_arriving_is_held_back_while_streaming():
    assert streaming_safe("Her mobile is 98412") == "Her mobile is "
    assert streaming_safe("Her mobile is +977-98") == "Her mobile is "
    assert streaming_safe("The rent is due monthly.") == "The rent is due monthly."


def test_masking_keeps_separators_and_a_few_digits():
    assert mask_value("card", "4111-1111-1111-1111") == "••••-••••-••••-1111"
    assert mask_value("phone", "9841234567") == "98••••••67"


def test_values_a_line_or_two_below_their_label_on_a_form():
    form = ("2b Employer Identification Number (EIN)\nAcme Corp software   (Do not enter your Social Security Number)\n"
            "Trade name of business (if different from name of employer)   735268329\n"
            "2c Employer's telephone number\nIn care of name   011536259\n2d Business code (see instructions)")
    found = [(f.kind, form[f.start:f.end]) for f in find_private(form)]
    assert found == [("id_number", "735268329"), ("phone", "011536259")]


def test_a_label_does_not_reach_past_other_numbers_or_take_a_year_range():
    assert find_private("Contact the archive about the records from 1939-1945.") == []
    assert find_private("Phone the office in room 12 before 3 pm; the reference is 20251234.") == []
